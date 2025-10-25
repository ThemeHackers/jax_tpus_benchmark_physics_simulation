import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint
import json
import time

try:
    jax.config.update('jax_platform_name', 'tpu')
    rprint("[green]TPU platform configured.[/green]")
except Exception as e:
    rprint(f"[yellow]Could not configure TPU, falling back to default. Error: {e}[/yellow]")

console = Console()

G = 1.0
c = 1.0
ISCO_FACTOR = 6.0

rprint("[bold blue]=== N-Body Black Hole Merger Simulator (3-Body Enhanced with TPU) ===[/bold blue]")
n_bodies = IntPrompt.ask("Number of black holes (2-5 recommended)", default=3)
masses = []
for i in range(n_bodies):
    m = FloatPrompt.ask(f"Mass of BH{i+1} (M☉)", default=30.0)
    masses.append(m)
initial_distance = FloatPrompt.ask("Typical initial separation", default=100.0)
initial_velocity = FloatPrompt.ask("Typical initial velocity (v/c)", default=0.1)
sim_time = FloatPrompt.ask("Simulation time", default=200.0)
D_gw = FloatPrompt.ask("GW observer distance (Mpc)", default=410.0)
compute_chaos = Prompt.ask("Compute Lyapunov exponent for chaos? (y/n)", default="y") == "y"

masses_np = np.array(masses)
M_total = np.sum(masses_np)

table = Table(title="N-Body Parameters")
table.add_column("Body", style="cyan")
table.add_column("Mass (M☉)", style="magenta")
for i, m in enumerate(masses_np):
    table.add_row(f"BH{i+1}", f"{m}")
table.add_row("Total Mass", f"{M_total}")
console.print(table)

masses_jnp = jnp.array(masses)

@jit
def pairwise_forces(positions, masses):
    n = len(positions)
    acc = jnp.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r_vec = positions[j] - positions[i]
            r_norm = jnp.linalg.norm(r_vec)
            
            acc_mag = jnp.where(r_norm >= 1e-6, G * masses[j] / r_norm**3, 0.0)
            acc = acc.at[i].add(acc_mag * r_vec)
            
    return acc

@jit
def nbody_ode(t, y, masses):
    n = len(masses)
    positions = jnp.reshape(y[:2*n], (n, 2))
    velocities = jnp.reshape(y[2*n:], (n, 2))
    
    acc = pairwise_forces(positions, masses)
    dydt = jnp.concatenate([velocities.flatten(), acc.flatten()])
    return dydt

@jit
def rk4_step(y, t, dt, masses):
    k1 = nbody_ode(t, y, masses)
    k2 = nbody_ode(t + 0.5 * dt, y + 0.5 * dt * k1, masses)
    k3 = nbody_ode(t + 0.5 * dt, y + 0.5 * dt * k2, masses)
    k4 = nbody_ode(t + dt, y + dt * k3, masses)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate_nbody(y0, masses, t0, tf, num_steps):
    dt = (tf - t0) / num_steps
    
    def body(carry, i):
        t_step = t0 + i * dt
        y_new = rk4_step(carry, t_step, dt, masses)
        return y_new, y_new
    
    xs = jnp.arange(num_steps)
    _, ys = lax.scan(body, y0, xs)
    return jnp.vstack([y0[None, :], ys])

def init_state(n, dist, vel):
    y0 = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = dist * np.cos(angle) / 2
        y = dist * np.sin(angle) / 2
        vx = -vel * np.sin(angle)
        vy = vel * np.cos(angle)
        y0.extend([x, y, vx, vy])
    return np.array(y0)

initial_state_np = init_state(n_bodies, initial_distance, initial_velocity)
initial_state = jnp.array(initial_state_np)
t_span = (0, sim_time)
num_steps = 1000

simulate_nbody_jitted = jit(simulate_nbody, static_argnums=(4,))

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task_warmup = progress.add_task("Compiling JAX/TPU code (warm-up)...", total=1)
    _ = simulate_nbody_jitted(initial_state, masses_jnp, t_span[0], t_span[1], num_steps).block_until_ready()
    progress.update(task_warmup, advance=1)
    progress.remove_task(task_warmup)

    task_run = progress.add_task("Simulating N-body dynamics on TPU...", total=1)
    
    start_time = time.perf_counter()
    
    sol_y = simulate_nbody_jitted(initial_state, masses_jnp, t_span[0], t_span[1], num_steps)
    sol_y.block_until_ready()
    
    end_time = time.perf_counter()
    sim_duration = end_time - start_time
    
    sol_t = jnp.linspace(t_span[0], t_span[1], num_steps + 1)
    progress.update(task_run, advance=1)

rprint(f"[bold cyan]TPU Simulation (Main) took: {sim_duration * 1000:.2f} ms ({sim_duration:.4f} s)[/bold cyan]")

class Solution:
    def __init__(self, t, y):
        self.t = np.array(t)
        self.y = np.array(y)

sol = Solution(sol_t, sol_y)

positions_t = np.array([sol.y[i, :2*n_bodies].reshape(n_bodies, 2) for i in range(len(sol.t))])

def multi_gw_strain(t, positions_t, masses, D_gw):
    h_plus = np.zeros_like(t)
    n_pairs = 0
    D_gw_meters = D_gw * 3.086e22
    
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            n_pairs += 1
            r_ij = np.linalg.norm(positions_t[:, i] - positions_t[:, j], axis=1)
            r_ij[r_ij < 1e-6] = 1e-6 
            
            mu_ij = masses[i] * masses[j] / (masses[i] + masses[j])
            chirp_ij = mu_ij ** (3./5) * (masses[i] + masses[j]) ** (2./5)
            
            omega_ij = np.sqrt(G * (masses[i] + masses[j]) / r_ij**3)
            
            dt = np.diff(t, prepend=t[0])
            phi_ij = np.zeros_like(t)
            phi_ij[1:] = np.cumsum(omega_ij[1:] * dt[1:])
            
            amp_ij = (4 * (G * chirp_ij)**(5/3) / (c**4 * D_gw_meters)) * (omega_ij)**(2/3)
            
            h_plus += amp_ij * np.cos(2 * phi_ij)
            
    return h_plus / max(n_pairs, 1)

h_plus = multi_gw_strain(sol.t, positions_t, masses_np, D_gw)

if compute_chaos:
    d0 = 1e-6
    
    pert_state = initial_state.at[0].add(d0)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Simulating perturbed trajectory on TPU...", total=1)
        
        start_pert_time = time.perf_counter()
        
        sol_pert_y = simulate_nbody_jitted(pert_state, masses_jnp, t_span[0], t_span[1], num_steps)
        sol_pert_y.block_until_ready()
        
        end_pert_time = time.perf_counter()
        pert_duration = end_pert_time - start_pert_time
        
        progress.update(task, advance=1)
    
    rprint(f"[bold cyan]TPU Simulation (Perturbed) took: {pert_duration * 1000:.2f} ms ({pert_duration:.4f} s)[/bold cyan]")
    
    sol_pert = Solution(sol_t, sol_pert_y)
    
    delta = np.linalg.norm(sol.y - sol_pert.y, axis=1)
    
    valid_indices = (sol.t > 1e-10) & (delta > 1e-15)
    
    if np.sum(valid_indices) > 0:
        t_valid = sol.t[valid_indices]
        delta_valid = delta[valid_indices]
        
        lyap_exp = np.mean(np.log(delta_valid / d0) / t_valid)
        rprint(f"[yellow]Lyapunov Exponent: {lyap_exp:.3f} (positive = chaotic orbit!)[/yellow]")
    else:
        rprint("[yellow]Could not compute Lyapunov Exponent (no divergence detected).[/yellow]")

colors = plt.cm.Set1(np.linspace(0, 1, n_bodies))

fig_gw, ax_gw = plt.subplots(figsize=(10, 4))
ax_gw.plot(sol.t, h_plus, label='Multi-Body h+', color='red')
ax_gw.set_xlabel('Time'); ax_gw.set_ylabel('Strain')
ax_gw.set_title('N-Body Gravitational Waveform')
ax_gw.legend(); ax_gw.grid(True)
plt.savefig('n_body_gw_plot.png')
rprint("[green]GW plot saved: n_body_gw_plot.png[/green]")

sample_rate = 44100
gw_normalized = h_plus / (np.max(np.abs(h_plus)) + 1e-10)
gw_boosted = gw_normalized * 5.0
gw_clipped = np.clip(gw_boosted, -1.0, 1.0)
gw_audio = np.int16(gw_clipped * 32767)

wavfile.write('n_body_gw.wav', sample_rate, gw_audio)
rprint("[green]Multi-body GW audio: n_body_gw.wav (listen for multiple chirps!)[/green]")

fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
for i in range(n_bodies):
    ax_3d.plot(positions_t[:, i, 0], positions_t[:, i, 1], sol.t, color=colors[i], label=f'BH{i+1}')
ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Time')
ax_3d.set_title('N-Body (e.g., 3-Body) Trajectories')
ax_3d.legend()
plt.savefig('n_body_3d_plot.png')
rprint("[green]3D plot saved: n_body_3d_plot.png[/green]")

all_x = positions_t[..., 0].flatten()
all_y = positions_t[..., 1].flatten()
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
padding_x = (x_max - x_min) * 0.1
padding_y = (y_max - y_min) * 0.1
x_lim = [x_min - padding_x, x_max + padding_x]
y_lim = [y_min - padding_y, y_max + padding_y]

fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def animate(frame):
    ax1.clear(); ax2.clear()
    idx = min(frame * 2, len(sol.t)-1)
    
    for i in range(n_bodies):
        ax1.plot(positions_t[:idx+1, i, 0], positions_t[:idx+1, i, 1], color=colors[i], alpha=0.4, lw=1)
    
    pos_frame = positions_t[idx]
    ax1.scatter(pos_frame[:, 0], pos_frame[:, 1], c=colors[:n_bodies], s=100, zorder=10)
    
    ax1.set_xlim(x_lim); ax1.set_ylim(y_lim)
    ax1.set_title(f'N-Body Positions at t={sol.t[idx]:.1f}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    
    ax2.plot(sol.t[:idx+1], h_plus[:idx+1], 'r-')
    ax2.set_title('Cumulative GW')
    ax2.set_xlabel('Time'); ax_gw.set_ylabel('Strain'); ax2.grid(True)
    
    h_min, h_max = np.min(h_plus), np.max(h_plus)
    h_pad = (h_max - h_min) * 0.1
    ax2.set_ylim(h_min - h_pad, h_max + h_pad)
    
    return []

n_frames = (len(sol.t) // 2) + 1
anim = FuncAnimation(fig_anim, animate, frames=n_frames, interval=100, blit=False)
writer = PillowWriter(fps=10)

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task = progress.add_task("Generating animation (n_body_merger.gif)...", total=n_frames)
    anim.save('n_body_merger.gif', writer=writer, progress_callback=lambda i, n: progress.update(task, advance=1))

rprint("[green]N-Body GIF: n_body_merger.gif[/green]")

data = {'t': sol.t.tolist(), 'positions': positions_t.tolist(), 'h_plus': h_plus.tolist()}
with open('n_body_data.json', 'w') as f:
    json.dump(data, f)
rprint("[green]Data: n_body_data.json[/green]")

rprint("[bold green]N-Body Simulation complete on TPU! For 3-body, expect chaotic paths. 🚀 Enhanced efficiency via TPU ODE integration[/bold green]")

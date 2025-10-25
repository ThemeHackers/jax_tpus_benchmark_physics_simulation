import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.integrate import solve_ivp
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

console = Console()

G = 1.0
c = 1.0
ISCO_FACTOR = 6.0

print("[bold blue]=== N-Body Black Hole Merger Simulator (3-Body Enhanced) ===[/bold blue]")
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

masses = np.array(masses)
M_total = np.sum(masses)

table = Table(title="N-Body Parameters")
table.add_column("Body", style="cyan")
table.add_column("Mass (M☉)", style="magenta")
for i, m in enumerate(masses):
    table.add_row(f"BH{i+1}", f"{m}")
table.add_row("Total Mass", f"{M_total}")
console.print(table)

@jit
def pairwise_forces(positions, masses):
    n = len(positions)
    acc = jnp.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r_vec = positions[j] - positions[i]
            r_norm = jnp.linalg.norm(r_vec)
            if r_norm < 1e-6: continue
            force_mag = G * masses[i] * masses[j] / r_norm**3
            acc = acc.at[i].add(force_mag * r_vec)
    return acc

def nbody_ode(t, y, masses):
    n = len(masses)
    positions = jnp.reshape(y[:2*n], (n, 2))
    velocities = jnp.reshape(y[2*n:], (n, 2))
    
    for i in range(n):
        for j in range(i+1, n):
            r_ij = jnp.linalg.norm(positions[i] - positions[j])
            if r_ij < ISCO_FACTOR * (masses[i] + masses[j]):
                pass
    
    acc = pairwise_forces(positions, masses)
    dydt = jnp.concatenate([velocities.flatten(), acc.flatten()])
    return dydt

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

initial_state = init_state(n_bodies, initial_distance, initial_velocity)
t_span = (0, sim_time)

sol = solve_ivp(fun=lambda t, y: np.array(nbody_ode(t, y, masses)), t_span=t_span, y0=initial_state,
                method='RK45', rtol=1e-6, max_step=1.0)

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task = progress.add_task("Simulating N-body dynamics...", total=len(sol.t))
    for i in range(1, len(sol.t)):
        progress.update(task, advance=1)

positions_t = np.array([sol.y[:2*n_bodies, i].reshape(n_bodies, 2) for i in range(len(sol.t))])

def multi_gw_strain(t, positions_t, masses, D_gw):
    h_plus = np.zeros_like(t)
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_ij = np.linalg.norm(positions_t[:, i] - positions_t[:, j], axis=1)
            mu_ij = masses[i] * masses[j] / (masses[i] + masses[j])
            chirp_ij = (mu_ij * (masses[i] + masses[j]) / mu_ij)**(3/5) * mu_ij**(3/5)
            omega_ij = np.sqrt(G * (masses[i] + masses[j]) / r_ij**3)
            phi_ij = np.cumsum(omega_ij) * np.diff(t, prepend=0)
            amp_ij = (4 * (G * chirp_ij)**(5/3) / (c**4 * D_gw * 3.086e22)) * (np.pi * np.abs(t) * omega_ij / 1)**(2/3)
            h_plus += amp_ij * np.cos(2 * phi_ij)
    return h_plus / n_bodies

h_plus = multi_gw_strain(sol.t, positions_t, masses, D_gw)

if compute_chaos:
    pert_state = initial_state + 1e-6
    sol_pert = solve_ivp(fun=lambda t, y: np.array(nbody_ode(t, y, masses)), t_span=t_span, y0=pert_state,
                         method='RK45', rtol=1e-6)
    delta = np.linalg.norm(sol.y - sol_pert.y, axis=0)
    lyap_exp = np.mean(np.log(delta[1:] / 1e-6) / sol.t[1:])
    rprint(f"[yellow]Lyapunov Exponent: {lyap_exp:.3f} (positive = chaotic orbit!)[/yellow]")

fig_gw, ax_gw = plt.subplots(figsize=(10, 4))
ax_gw.plot(sol.t, h_plus, label='Multi-Body h+', color='red')
ax_gw.set_xlabel('Time'); ax_gw.set_ylabel('Strain')
ax_gw.set_title('N-Body Gravitational Waveform')
ax_gw.legend(); ax_gw.grid(True)
plt.show()

sample_rate = 44100
gw_audio = np.int16(h_plus / np.max(np.abs(h_plus)) * 32767 * 10)
wavfile.write('n_body_gw.wav', sample_rate, gw_audio)
rprint("[green]Multi-body GW audio: n_body_gw.wav (listen for multiple chirps!)[/green]")

fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
colors = plt.cm.Set1(np.linspace(0, 1, n_bodies))
for i in range(n_bodies):
    ax_3d.plot(positions_t[:, i, 0], positions_t[:, i, 1], sol.t, color=colors[i], label=f'BH{i+1}')
ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Time')
ax_3d.set_title('N-Body (e.g., 3-Body) Trajectories')
ax_3d.legend()
plt.show()

fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
x_lim = [-150, 150]; y_lim = [-150, 150]
def animate(frame):
    ax1.clear(); ax2.clear()
    idx = min(frame * 2, len(sol.t)-1)
    pos_frame = positions_t[idx]
    ax1.scatter(pos_frame[:, 0], pos_frame[:, 1], c=colors[:n_bodies], s=100)
    ax1.set_xlim(x_lim); ax1.set_ylim(y_lim)
    ax1.set_title(f'N-Body Positions at t={sol.t[idx]:.1f}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    
    ax2.plot(sol.t[:idx+1], h_plus[:idx+1], 'r-')
    ax2.set_title('Cumulative GW')
    ax2.set_xlabel('Time'); ax2.set_ylabel('Strain'); ax2.grid(True)
    return []

anim = FuncAnimation(fig_anim, animate, frames=len(sol.t)//2, interval=100, blit=False)
writer = PillowWriter(fps=10)
anim.save('n_body_merger.gif', writer=writer)
rprint("[green]N-Body GIF: n_body_merger.gif[/green]")

data = {'t': sol.t.tolist(), 'positions': positions_t.tolist(), 'h_plus': h_plus.tolist()}
with open('n_body_data.json', 'w') as f:
    json.dump(data, f)
rprint("[green]Data: n_body_data.json[/green]")

rprint("[bold green]N-Body Simulation complete! For 3-body, expect chaotic paths. 🚀[/bold green]")

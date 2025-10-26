import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.lax import fori_loop
import optax
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio
from scipy.stats import norm
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
import argparse

def main(args):
    console = Console()

    N_WALKERS = args.n_walkers
    N_EPOCHS = args.n_epochs
    N_EQUIL_STEPS = args.n_equil
    STEP_SIZE = args.step_size
    LEARNING_RATE = args.lr
    N_DMC_STEPS = args.n_dmc
    DMC_DT = args.dmc_dt
    DIM = args.dim

    def potential_energy(x):
        return 0.5 * jnp.sum(x**2, axis=-1)

    def log_psi_fun(x, alpha):
        return -alpha * jnp.sum(x**2, axis=-1)

    def kinetic_energy(x, alpha):
        r2 = jnp.sum(x**2, axis=-1)
        laplacian_log_psi = -2.0 * alpha * DIM
        grad_log_psi_norm_sq = 4.0 * alpha**2 * r2
        ke = -0.5 * (laplacian_log_psi + grad_log_psi_norm_sq)
        return ke

    @jit
    def local_energy(x, alpha):
        ke = kinetic_energy(x, alpha)
        pe = potential_energy(x)
        return ke + pe

    grad_log_psi_alpha = jit(grad(log_psi_fun, argnums=1))

    @jit
    def metropolis_step(walker, alpha, key):
        key, subkey = random.split(key)
        proposed_walker = walker + STEP_SIZE * random.uniform(subkey, shape=walker.shape, minval=-0.5, maxval=0.5)
        
        log_prob_new = 2.0 * log_psi_fun(proposed_walker, alpha)
        log_prob_old = 2.0 * log_psi_fun(walker, alpha)
        
        key, subkey = random.split(key)
        acceptance_prob = jnp.exp(log_prob_new - log_prob_old)
        accept = random.uniform(subkey, shape=()) < acceptance_prob
        
        new_walker = jnp.where(accept, proposed_walker, walker)
        
        return new_walker, key

    vmap_metropolis_step = vmap(metropolis_step, in_axes=(0, None, 0))

    @jit
    def vmc_epoch_step(state):
        walkers, alpha, key, opt_state = state
        
        def equil_loop_body(i, loop_state):
            walkers, key = loop_state
            keys_for_walkers = random.split(key, N_WALKERS)
            new_walkers, _ = vmap_metropolis_step(walkers, alpha, keys_for_walkers)
            return (new_walkers, keys_for_walkers[0])
        
        key, subkey = random.split(key)
        (walkers, key) = fori_loop(0, N_EQUIL_STEPS, equil_loop_body, (walkers, subkey))

        vmap_E_L = vmap(local_energy, in_axes=(0, None))
        energies = vmap_E_L(walkers, alpha)
        E_mean = jnp.mean(energies)
        
        vmap_grad_log = vmap(grad_log_psi_alpha, in_axes=(0, None))
        log_psi_grads = vmap_grad_log(walkers, alpha)
        
        grad_E = 2.0 * jnp.mean((energies - E_mean) * log_psi_grads)
        
        updates, new_opt_state = optimizer.update(grad_E, opt_state)
        new_alpha = optax.apply_updates(alpha, updates)
        
        new_alpha = jnp.maximum(0.01, new_alpha) 
        
        new_state = (walkers, new_alpha, key, new_opt_state)
        return new_state, E_mean, grad_E

    console.print(Rule(f"[bold cyan]Variational Monte Carlo (VMC) Simulation for {DIM}D[/bold cyan]"))
    param_str = (
        f"[bold]N_WALKERS[/bold] = [magenta]{N_WALKERS}[/magenta]\n"
        f"[bold]DIM[/bold]        = [magenta]{DIM}[/magenta]\n"
        f"[bold]N_EPOCHS[/bold]  = [magenta]{N_EPOCHS}[/magenta]\n"
        f"[bold]N_EQUIL[/bold]   = [magenta]{N_EQUIL_STEPS}[/magenta]\n"
        f"[bold]STEP_SIZE[/bold] = [magenta]{STEP_SIZE}[/magenta]\n"
        f"[bold]LR[/bold]        = [magenta]{LEARNING_RATE}[/magenta]"
    )
    console.print(Panel(param_str, title="VMC Parameters", expand=False))

    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    walkers = random.normal(subkey, shape=(N_WALKERS, DIM))
    alpha_init = 1.0

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(alpha_init)

    energies_history = []
    alpha_history = []

    vmc_frame_files = []
    VMC_FRAMES_DIR = "vmc_frames"
    if not args.no_gif:
        os.makedirs(VMC_FRAMES_DIR, exist_ok=True)
        
    x_theory = np.linspace(-3.5, 3.5, 300)
    sigma = 1 / np.sqrt(2)
    psi_sq_theory = norm.pdf(x_theory, 0, sigma)
    psi_sq_theory /= np.trapezoid(psi_sq_theory, x_theory)

    current_state = (walkers, alpha_init, key, opt_state)

    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]

    with Progress(*progress_columns, console=console, transient=False) as progress:
        vmc_task = progress.add_task("[red]VMC Optimization...", total=N_EPOCHS)

        for epoch in range(N_EPOCHS):
            current_state, energy, grad_value = vmc_epoch_step(current_state) 
            
            energies_history.append(energy)
            alpha_history.append(current_state[1])
            
            progress.update(
                vmc_task, 
                advance=1, 
                description=f"[red]Epoch {epoch:4d} | E={energy:8.6f} | α={current_state[1]:.6f}"
            )

            if not args.no_gif and epoch % 5 == 0:
                filename = f"{VMC_FRAMES_DIR}/vmc_frame_{epoch:04d}.png"
                vmc_frame_files.append(filename)
                
                plt.figure(figsize=(10, 6))
                plt.hist(np.array(current_state[0][:, 0]), bins=50, density=True, label=f'VMC Walkers (Epoch {epoch})', alpha=0.7)
                plt.plot(x_theory, psi_sq_theory, 'r-', linewidth=2, label='Exact Marginal $|\Psi_0|^2$ (x-coordinate)')
                plt.title(f'VMC Epoch {epoch:04d} | $\\alpha$ = {current_state[1]:.4f} | E = {energy:.4f}')
                plt.xlabel('Position (x)')
                plt.ylabel('Probability Density $|\Psi(x)|^2$')
                plt.legend()
                plt.xlim(-3.5, 3.5)
                plt.ylim(0, 0.8)
                plt.savefig(filename)
                plt.close()

    console.print(Rule("[bold green]VMC Simulation Finished[/bold green]"))
    E_analytical = DIM * 0.5
    console.print(f"Analytical solution: E_0 = {E_analytical}, alpha = 0.5")
    console.print(f"VMC result:          [bold green]E_0 = {energies_history[-1]:.6f}, alpha = {alpha_history[-1]:.6f}[/bold green]")

    if not args.no_gif:
        with console.status("[yellow]Creating VMC GIF (vmc_animation.gif)...") as status:
            with imageio.get_writer('vmc_animation.gif', mode='I', duration=0.1, loop=0) as writer:
                for filename in vmc_frame_files:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        console.print("[green]VMC GIF finished![/green]")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(energies_history, label='VMC Energy')
    ax1.axhline(y=E_analytical, color='r', linestyle='--', label=f'Exact $E_0 = {E_analytical}$')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.set_title(f'VMC Optimization of {DIM}D Quantum Harmonic Oscillator')
    ax2.plot(alpha_history, label='$\\alpha$ value')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Exact $\\alpha = 0.5$')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Parameter $\\alpha$')
    ax2.legend()
    plt.tight_layout()
    if not args.no_plot:
        plt.show()

    x_theory_final_plot = jnp.linspace(-3, 3, 200)
    psi_sq_theory_final_plot = jnp.exp(-x_theory_final_plot**2)
    psi_sq_theory_final_plot /= np.trapezoid(psi_sq_theory_final_plot, x_theory_final_plot)

    final_walkers_vmc = current_state[0]
    final_alpha_vmc = current_state[1]

    plt.figure(figsize=(10, 6))
    plt.hist(np.array(final_walkers_vmc[:, 0]), bins=50, density=True, label=f'VMC Walkers (alpha={final_alpha_vmc:.3f})')
    plt.plot(x_theory_final_plot, psi_sq_theory_final_plot, 'r-', label='Exact Marginal $|\Psi_0|^2$ (x-coordinate)')
    plt.title('Final VMC Walker Distribution vs. Exact Ground State Marginal')
    plt.xlabel('Position (x)')
    plt.ylabel('Probability Density $|\Psi(x)|^2$')
    plt.legend()
    if not args.no_plot:
        plt.show()

    console.print(Rule(f"[bold cyan]Diffusion Monte Carlo (DMC) Simulation for {DIM}D[/bold cyan]"))

    ALPHA_BEST = alpha_history[-1]
    WALKERS_DMC_INIT = current_state[0]
    key = current_state[2]

    param_str_dmc = (
        f"[bold]N_DMC_STEPS[/bold] = [magenta]{N_DMC_STEPS}[/magenta]\n"
        f"[bold]DMC_DT[/bold]      = [magenta]{DMC_DT}[/magenta]\n"
        f"[bold]ALPHA_BEST[/bold]  = [magenta]{ALPHA_BEST:.6f}[/magenta]"
    )
    console.print(Panel(param_str_dmc, title="DMC Parameters", expand=False))

    def grad_log_psi_x(x, alpha):
        return -2.0 * alpha * x

    drift_force_fun = lambda x, alpha: 2.0 * grad_log_psi_x(x, alpha)
    vmap_drift_force = vmap(drift_force_fun, in_axes=(0, None))

    vmap_E_L = vmap(local_energy, in_axes=(0, None))

    @jit
    def dmc_step_body(state, i):
        walkers, key = state
        
        E_local = vmap_E_L(walkers, ALPHA_BEST)
        E_ref = jnp.mean(E_local)
        
        weights = jnp.exp(-(E_local - E_ref) * DMC_DT)
        
        key, subkey = random.split(key)
        weights_normalized = weights / jnp.sum(weights)

        weights_normalized = jnp.nan_to_num(weights_normalized, nan=1e-9)
        weights_normalized = jnp.where(jnp.isinf(weights_normalized), 1e-9, weights_normalized)
        weights_sum = jnp.sum(weights_normalized)
        weights_normalized = jnp.where(weights_sum == 0, 1.0/N_WALKERS, weights_normalized / weights_sum)

        resampled_walkers = random.choice(
            subkey,
            a=walkers,
            shape=(N_WALKERS,),
            p=weights_normalized
        )

        key, key_drift, key_diff = random.split(key, 3)
        
        drift_force = vmap_drift_force(resampled_walkers, ALPHA_BEST)
        drift_move = drift_force * DMC_DT
        
        diffusion_move = random.normal(key_diff, shape=(N_WALKERS, DIM)) * jnp.sqrt(DMC_DT)
        
        new_walkers = resampled_walkers + drift_move + diffusion_move
        
        new_state = (new_walkers, key)
        
        return new_state, (new_walkers, E_ref) 

    with console.status("[yellow]Running JIT-compiled DMC steps (jax.lax.scan)...") as status:
        dmc_initial_state = (WALKERS_DMC_INIT, key)
        (final_dmc_state, (dmc_walkers_history, dmc_energy_history)) = jax.lax.scan(
            dmc_step_body, dmc_initial_state, jnp.arange(N_DMC_STEPS)
        )
    console.print("[green]...DMC scan finished.[/green]")

    dmc_frame_files = []
    DMC_FRAMES_DIR = "dmc_frames"
    if not args.no_gif:
        os.makedirs(DMC_FRAMES_DIR, exist_ok=True)

    if not args.no_gif:
        with Progress(*progress_columns, console=console, transient=False) as progress:
            dmc_gif_task = progress.add_task("[blue]Creating DMC GIF frames...", total=N_DMC_STEPS)
            for i in range(N_DMC_STEPS):
                progress.update(dmc_gif_task, advance=1)
                if i % 5 == 0:
                    filename = f"{DMC_FRAMES_DIR}/dmc_frame_{i:04d}.png"
                    dmc_frame_files.append(filename)
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(np.array(dmc_walkers_history[i][:, 0]), bins=50, density=True, label=f'DMC Walkers (Step {i})', alpha=0.7, color='green')
                    plt.plot(x_theory, psi_sq_theory, 'r-', linewidth=2, label='Exact Marginal $|\Psi_0|^2$')
                    plt.title(f'DMC Step {i:04d} | $E_{{ref}}$ = {dmc_energy_history[i]:.4f}')
                    plt.xlabel('Position (x)')
                    plt.ylabel('Probability Density $|\Psi(x)|^2$')
                    plt.legend()
                    plt.xlim(-3.5, 3.5)
                    plt.ylim(0, 0.8)
                    plt.savefig(filename)
                    plt.close()

    if not args.no_gif:
        with console.status("[yellow]Compiling DMC GIF (dmc_animation.gif)...") as status:
            with imageio.get_writer('dmc_animation.gif', mode='I', duration=0.05, loop=0) as writer:
                for filename in dmc_frame_files:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        console.print("[green]DMC GIF finished![/green]")

    plt.figure(figsize=(10, 6))
    burn_in = 100
    plt.plot(dmc_energy_history[burn_in:], label=f'DMC Energy ($E_{{ref}}$) after step {burn_in}')

    dmc_mean_energy = jnp.mean(dmc_energy_history[burn_in:])
    dmc_std_error = jnp.std(dmc_energy_history[burn_in:]) / jnp.sqrt(N_DMC_STEPS - burn_in)

    plt.axhline(y=dmc_mean_energy, color='b', linestyle='--', 
                label=f'DMC Mean = {dmc_mean_energy:.6f} $\\pm$ {dmc_std_error:.6f}')
    plt.axhline(y=E_analytical, color='r', linestyle=':', label=f'Exact $E_0 = {E_analytical}$')
    plt.xlabel('DMC Step')
    plt.ylabel('Energy ($E_{{ref}}$)')
    plt.title(f'DMC Ground State Energy for {DIM}D')
    plt.legend()
    if not args.no_plot:
        plt.show()

    plt.figure(figsize=(10, 6))
    final_dmc_walkers, _ = final_dmc_state

    plt.hist(np.array(final_walkers_vmc[:, 0]), bins=50, density=True, label=f'VMC Walkers (Final)', alpha=0.6)
    plt.hist(np.array(final_dmc_walkers[:, 0]), bins=50, density=True, label=f'DMC Walkers (Final)', alpha=0.6, color='green')

    plt.plot(x_theory_final_plot, psi_sq_theory_final_plot, 'r-', linewidth=2, label='Exact Marginal $|\Psi_0|^2$')
    plt.title('Final Walker Distribution Marginal (VMC vs DMC vs Exact)')
    plt.xlabel('Position (x)')
    plt.ylabel('Probability Density $|\Psi(x)|^2$')
    plt.legend()
    if not args.no_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX VMC+DMC for D-Dimensional Quantum Harmonic Oscillator")
    parser.add_argument('--n_walkers', type=int, default=10000, help='Number of walkers')
    parser.add_argument('--n_epochs', type=int, default=3000, help='Number of VMC optimization epochs')
    parser.add_argument('--n_equil', type=int, default=100, help='Number of equilibration steps per epoch')
    parser.add_argument('--step_size', type=float, default=2.0, help='Metropolis step size')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate for VMC optimizer')
    parser.add_argument('--n_dmc', type=int, default=500, help='Number of DMC steps')
    parser.add_argument('--dmc_dt', type=float, default=0.01, help='Time step for DMC')
    parser.add_argument('--dim', type=int, default=3, help='Dimension')
    parser.add_argument('--no-gif', action='store_true', help='Disable GIF generation')
    parser.add_argument('--no-plot', action='store_true', help='Disable showing matplotlib plots')
    
    args = parser.parse_args()
    main(args)

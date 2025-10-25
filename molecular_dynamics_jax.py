import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap

import matplotlib.pyplot as plt
import time
import argparse 

from rich import print
from rich.panel import Panel
from rich.table import Table

def main(args):
    
    dimension = 2
    N = args.N
    rho = args.rho
    kT = args.kT
    dt = args.dt
    equilibration_steps = args.eq_steps
    production_steps = args.prod_steps
    sample_every = args.sample_every
    key_seed = args.seed
    output_filename = args.output

    sigma = 1.0
    epsilon = 1.0
    
    key = random.PRNGKey(key_seed)
    box_size = jnp.sqrt(N / rho)
    volume = box_size ** dimension

    param_details = f"""
Particles (N): {N}
Density (rho): {rho:.2f}
Temperature (kT): {kT:.1f}
Box Size: {box_size:.2f} x {box_size:.2f}
Backend: [bold cyan]{jax.default_backend()}[/bold cyan]

Steps (Eq/Prod): {equilibration_steps:,} / {production_steps:,}
Time Step (dt): {dt}
PRNG Seed: {key_seed}
"""
    print(Panel(param_details, title="[bold]Molecular Dynamics Simulation[/bold]", subtitle="using JAX", expand=False))

    @jit
    def periodic_displacement(dr, box_size):
        return dr - box_size * jnp.round(dr / box_size)

    def total_energy_fn(R):
        dr_all_pairs = R[:, None, :] - R[None, :, :]
        dr_all_pairs = vmap(vmap(periodic_displacement, (0, None), 0), (0, None), 0)(dr_all_pairs, box_size) 
        r_sq_matrix = jnp.sum(dr_all_pairs**2, axis=-1)
        mask = jnp.logical_not(jnp.eye(N, dtype=bool))
        r_sq_matrix_safe = jnp.where(mask, r_sq_matrix, 1.0)
        sigma_over_r_sq = (sigma**2) / r_sq_matrix_safe
        sigma_over_r_6 = sigma_over_r_sq**3
        sigma_over_r_12 = sigma_over_r_6**2
        pair_energy = 4.0 * epsilon * (sigma_over_r_12 - sigma_over_r_6)
        pair_energy = jnp.where(mask, pair_energy, 0.0)
        total_energy = 0.5 * jnp.sum(pair_energy)
        return total_energy

    force_fn = jit(grad(lambda R: -total_energy_fn(R)))

    @jit
    def verlet_step(state):
        R, V = state
        F = force_fn(R)
        V_half = V + 0.5 * F * dt
        R_new = R + V_half * dt
        R_new = jnp.mod(R_new, box_size)
        F_new = force_fn(R_new)
        V_new = V_half + 0.5 * F_new * dt
        return (R_new, V_new)

    @jit
    def equilibrate_fn(initial_state):
        print("JIT compiling equilibration...") 
        def step_fn_wrapper(i, state):
            return verlet_step(state)
        final_state = jax.lax.fori_loop(0, equilibration_steps, step_fn_wrapper, initial_state)
        return final_state

    @jit
    def production_fn(initial_state):
        print("JIT compiling production...")
        num_samples = production_steps // sample_every 
        R_trajectory = jnp.zeros((num_samples, N, dimension))
        
        def step_fn_wrapper(i, carry):
            state, trajectory = carry
            state = verlet_step(state)
            idx = i // sample_every
            trajectory = jax.lax.cond(
                i % sample_every == 0,
                lambda traj: traj.at[idx].set(state[0]),
                lambda traj: traj,
                trajectory
            )
            return (state, trajectory)

        final_state, R_history = jax.lax.fori_loop(
            0, production_steps, step_fn_wrapper, (initial_state, R_trajectory)
        )
        return final_state, R_history

    def _calculate_g_r_internal(R_history, N_local, box_size_local, nbins, r_max):
        dr = r_max / nbins
        r_bins = jnp.linspace(0, r_max, nbins + 1)
        bin_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
        shell_volumes = jnp.pi * (r_bins[1:]**2 - r_bins[0:-1]**2)
        rho_pairs = (N_local * (N_local - 1) / 2.0) / (box_size_local**dimension)
        
        ideal_counts = rho_pairs * shell_volumes
        
        def get_histogram(R):
            dr_all_pairs = R[:, None, :] - R[None, :, :]
            dr_all_pairs = vmap(vmap(periodic_displacement, (0, None), 0), (0, None), 0)(dr_all_pairs, box_size_local)
            r_sq_matrix = jnp.sum(dr_all_pairs**2, axis=-1)
            indices = jnp.triu_indices(N_local, k=1)
            r_all_pairs_flat = jnp.sqrt(r_sq_matrix[indices])
            hist_counts, _ = jnp.histogram(r_all_pairs_flat, bins=r_bins)
            return hist_counts

        all_hists = jax.vmap(get_histogram)(R_history)
        avg_hist = jnp.mean(all_hists, axis=0)
        g_r = avg_hist / ideal_counts
        return bin_centers, g_r

    calculate_g_r = jit(_calculate_g_r_internal, static_argnums=(1, 3))

    key, r_key, v_key = random.split(key, 3)
    R_initial = random.uniform(r_key, (N, dimension)) * box_size
    V_initial = random.normal(v_key, (N, dimension)) * jnp.sqrt(kT) 
    state_initial = (R_initial, V_initial)

    print("\n:hourglass_flowing_sand: [bold yellow]--- Starting Equilibration ---[/bold yellow]")
    start_time_eq = time.time()
    

    state_eq = equilibrate_fn(state_initial) 


    state_eq[0].block_until_ready() 
    time_eq = time.time() - start_time_eq
    print(f":white_check_mark: Equilibration finished in [bold green]{time_eq:.2f} s[/bold green]")

    print("\n:movie_camera: [bold blue]--- Starting Production (sampling) ---[/bold blue]")
    start_time_prod = time.time()
    state_final, R_history = production_fn(state_eq)
    R_history.block_until_ready() 
    time_prod = time.time() - start_time_prod
    print(f":white_check_mark: Production finished in [bold green]{time_prod:.2f} s[/bold green]")

    print("\n:chart_increasing: [bold magenta]--- Calculating g(r) ---[/bold magenta]")
    dr_g = 0.05
    r_max_g = box_size / 2.0
    nbins_g = int(r_max_g / dr_g)

    start_time_g_r = time.time()
    r_bins_g, g_r = calculate_g_r(R_history, N, box_size, nbins_g, r_max_g)
    g_r.block_until_ready() 
    time_g_r = time.time() - start_time_g_r
    print(f":white_check_mark: g(r) calculation complete in [bold green]{time_g_r:.2f} s[/bold green]")

    summary_table = Table(title="[bold]Simulation Summary[/bold]")
    summary_table.add_column("Phase", style="cyan")
    summary_table.add_column("Time (s)", style="green")
    summary_table.add_column("Details", style="yellow")

    summary_table.add_row("Equilibration", f"{time_eq:.2f}", f"{equilibration_steps:,} steps")
    summary_table.add_row("Production", f"{time_prod:.2f}", f"{production_steps:,} steps")
    summary_table.add_row("g(r) Analysis", f"{time_g_r:.2f}", f"Collected {R_history.shape[0]} snapshots")
    summary_table.add_row("[bold]Total[/bold]", f"{time_eq + time_prod + time_g_r:.2f}", "")

    print(summary_table)

    print("\n:paintbrush: Displaying g(r) plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(r_bins_g, g_r, marker='o', markersize=4, linestyle='-')
    plt.title(f'Radial Distribution Function (g(r)) - PURE JAX - N={N}, rho={rho}, kT={kT}')
    plt.xlabel(r'Distance r (in units of $\sigma$)') 
    plt.ylabel('g(r)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(1.0, color='grey', linestyle='--')

    plt.savefig(output_filename, dpi=300, bbox_inches='tight') 
    print(f":floppy_disk: Plot saved as '{output_filename}'")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX Molecular Dynamics Simulation")
    
    parser.add_argument("--N", type=int, default=400, 
                        help="Number of particles (default: 400)")
    parser.add_argument("--rho", type=float, default=0.8, 
                        help="Density (default: 0.8)")
    parser.add_argument("--kT", type=float, default=1.0, 
                        help="Temperature (kT) (default: 1.0)")
    parser.add_argument("--dt", type=float, default=1e-3, 
                        help="Time step (default: 1e-3)")
    parser.add_argument("--eq_steps", type=int, default=10000, 
                        help="Equilibration steps (default: 10000)")
    parser.add_argument("--prod_steps", type=int, default=10000, 
                        help="Production steps (default: 10000)")
    parser.add_argument("--sample_every", type=int, default=100, 
                        help="Sample every N steps (default: 100)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="PRNG seed (default: 42)")
    parser.add_argument("--output", type=str, default="g_r_plot.png", 
                        help="Output plot filename (default: g_r_plot.png)")
    
    args = parser.parse_args()
    main(args)

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap

import matplotlib.pyplot as plt
import time

print(f"Running on: {jax.default_backend()}")

dimension = 2
N = 400
rho = 0.8
kT = 1.0

box_size = jnp.sqrt(N / rho)
volume = box_size ** dimension

print(f"Simulating {N} particles in box size {box_size:.2f}x{box_size:.2f} (density rho={rho:.2f})")

dt = 1e-3
equilibration_steps = 10000
production_steps = 10000
sample_every = 100

sigma = 1.0
epsilon = 1.0

key = random.PRNGKey(42)

@jit
def periodic_displacement(dr, box_size):
    return dr - box_size * jnp.round(dr / box_size)

def total_energy_fn(R, box_size):
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

force_fn = jit(grad(lambda R: -total_energy_fn(R, box_size)))

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
        0, 
        production_steps, 
        step_fn_wrapper, 
        (initial_state, R_trajectory)
    )
    return final_state, R_history

@jit
def calculate_g_r(R_history, N, box_size, nbins, r_max):
    dr = r_max / nbins
    r_bins = jnp.linspace(0, r_max, nbins + 1)
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    
    shell_volumes = jnp.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    
    rho_pairs = (N * (N - 1) / 2.0) / (box_size**dimension)
    
    ideal_counts = rho_pairs * shell_volumes
    
    def get_histogram(R):
        dr_all_pairs = R[:, None, :] - R[None, :, :]
        dr_all_pairs = vmap(vmap(periodic_displacement, (0, None), 0), (0, None), 0)(dr_all_pairs, box_size)
        r_sq_matrix = jnp.sum(dr_all_pairs**2, axis=-1)
        
        indices = jnp.triu_indices(N, k=1)
        r_all_pairs_flat = jnp.sqrt(r_sq_matrix[indices])
        
        hist_counts, _ = jnp.histogram(r_all_pairs_flat, bins=r_bins)
        return hist_counts

    all_hists = jax.vmap(get_histogram)(R_history)
    
    avg_hist = jnp.mean(all_hists, axis=0)
    
    g_r = avg_hist / ideal_counts
    
    return bin_centers, g_r

# We still need quantity.maxwell_boltzmann from jax_md for this example
# If you truly want zero jax_md, replace V_initial line with:
# V_initial = random.normal(v_key, (N, dimension)) * jnp.sqrt(kT)
# But we will import it just for this function for a correct simulation
from jax_md import quantity

key, r_key, v_key = random.split(key, 3)
R_initial = random.uniform(r_key, (N, dimension)) * box_size
V_initial = quantity.maxwell_boltzmann(v_key, N, dimension, kT) 

state_initial = (R_initial, V_initial)

print("--- Starting Equilibration ---")
start_time = time.time()
state_eq = equilibrate_fn(state_initial)
print(f"Equilibration (JIT) finished in {time.time() - start_time:.2f} s")

print("--- Starting Production (sampling) ---")
start_time = time.time()
state_final, R_history = production_fn(state_eq)
print(f"Production (JIT) finished in {time.time() - start_time:.2f} s")
print(f"Collected {R_history.shape[0]} snapshots")

print("Calculating g(r) (JAX from scratch)...")
dr_g = 0.05
r_max_g = box_size / 2.0
nbins_g = int(r_max_g / dr_g)

r_bins_g, g_r = calculate_g_r(R_history, N, box_size, nbins_g, r_max_g)
print("g(r) calculation complete")

plt.figure(figsize=(10, 6))
plt.plot(r_bins_g, g_r, marker='o', markersize=4, linestyle='-')
plt.title(f'Radial Distribution Function (g(r)) - PURE JAX - N={N}, rho={rho}, kT={kT}')
plt.xlabel(r'Distance r (in units of $\sigma$)')
plt.ylabel('g(r)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(1.0, color='grey', linestyle='--')
plt.show()

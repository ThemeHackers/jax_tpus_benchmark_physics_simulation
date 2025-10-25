import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--G', type=float, default=1.0)
parser.add_argument('--Bz', type=float, default=1.0)
parser.add_argument('--Bk', type=float, default=0.0)
parser.add_argument('--Ex', type=float, default=0.0)
parser.add_argument('--Ey', type=float, default=0.0)
args = parser.parse_args()

G = args.G
dt = args.dt
n_steps = args.n_steps

def acceleration(pos, vel, masses, charges):
    def pairwise_acc(i):
        ri = pos[i]
        acc_grav = jnp.zeros(2)
        for j in range(len(pos)):
            if i == j:
                continue
            rj = pos[j]
            r_diff = rj - ri
            r_norm = jnp.linalg.norm(r_diff)
            if r_norm < 1e-6:
                r_norm = 1e-6
            acc_grav += G * masses[j] * r_diff / (r_norm ** 3)
        return acc_grav
    acc_grav = vmap(pairwise_acc)(jnp.arange(len(pos)))
    
    def mag_acc(i):
        qm = charges[i] / masses[i]
        vx, vy = vel[i]
        x, y = pos[i]
        bz = args.Bz + args.Bk * x
        return qm * jnp.array([vy * bz, -vx * bz])
    acc_mag = vmap(mag_acc)(jnp.arange(len(pos)))
    
    def elec_acc(i):
        qm = charges[i] / masses[i]
        return qm * jnp.array([args.Ex, args.Ey])
    acc_elec = vmap(elec_acc)(jnp.arange(len(pos)))
    
    return acc_grav + acc_mag + acc_elec

@jit
def step(pos, vel, masses, charges):
    acc = acceleration(pos, vel, masses, charges)
    vel_new = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_new
    acc_new = acceleration(pos_new, vel_new, masses, charges)
    vel_new = vel_new + 0.5 * dt * acc_new
    return pos_new, vel_new

def simulate(pos_init, vel_init, masses, charges):
    pos = pos_init
    vel = vel_init
    trajectory = [pos.copy()]
    for _ in range(n_steps):
        pos, vel = step(pos, vel, masses, charges)
        trajectory.append(pos.copy())
    return jnp.array(trajectory)

n_particles = 3
pos_init = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
vel_init = jnp.array([[0.0, 0.1], [0.0, -0.1], [-0.1, 0.0]])
masses = jnp.array([1.0, 1.0, 1.0])
charges = jnp.array([1.0, 1.0, 1.0])

trajectory = simulate(pos_init, vel_init, masses, charges)

fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
lines = [ax.plot([], [], 'o')[0] for _ in range(n_particles)]

def animate(frame):
    for i, line in enumerate(lines):
        line.set_data(trajectory[frame][i, 0], trajectory[frame][i, 1])
    return lines

anim = FuncAnimation(fig, animate, frames=n_steps, interval=20, blit=True)
anim.save('three_particles_em_nonuni.gif', writer='pillow')
plt.show()

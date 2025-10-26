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

    r_diff = pos[None, :, :] - pos[:, None, :]
    

    r_norm_sq = jnp.sum(r_diff**2, axis=-1) + jnp.eye(len(pos))
    
    r_norm_sq_safe = jnp.where(r_norm_sq < 1e-12, 1e-12, r_norm_sq)

    r_norm_inv_cubed = r_norm_sq_safe**(-1.5)
    
    acc_grav_pairs = G * masses[None, :, None] * r_diff * r_norm_inv_cubed[..., None]
    
   
    acc_grav = jnp.sum(acc_grav_pairs, axis=1) 
    
    qm = (charges / masses)[:, None]     
    vx, vy = vel[:, 0], vel[:, 1]        
    x = pos[:, 0]                       
    
    bz = args.Bz + args.Bk * x           
    
    acc_mag_x = qm[:, 0] * vy * bz
    acc_mag_y = qm[:, 0] * -vx * bz
    acc_mag = jnp.stack([acc_mag_x, acc_mag_y], axis=1) 

    E_field = jnp.array([args.Ex, args.Ey])  
    acc_elec = qm * E_field[None, :]            
    
    return acc_grav + acc_mag + acc_elec
    
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
lines, = ax.plot([], [], 'o')


def animate(frame):
    
    x_data = trajectory[frame, :, 0]
    y_data = trajectory[frame, :, 1]
    
    lines.set_data(x_data, y_data)
    
    return lines,

anim = FuncAnimation(fig, animate, frames=n_steps, interval=20, blit=True)
anim.save('three_particles_em_nonuni.gif', writer='pillow')
plt.show()

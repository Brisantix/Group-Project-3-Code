#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 21:57:00 2025

@author: menglingshu
"""

#Traffic light problem in PDE model
import numpy as np
import matplotlib.pyplot as plt


# 1. parameter

L = 1000.0       # road length
N_x = 200        # grid number 
T = 60.0         # simulating time
dt = 0.05        # timestep
rho_max = 0.25   # max density
v_max = 30.0     # max speed
dx = L / N_x
time_steps = int(T / dt)

x = np.linspace(0, L, N_x)

#two kind of velocity relationship
def v1(rho, v_lim):
    return v_max * (1 - rho / rho_max)

def v2(rho, v_lim):

    return v_max * ((1 - rho / rho_max)**2)

def flux1(rho, v_lim):
    return rho * v1(rho, v_lim)

def flux2(rho, v_lim):
    return rho * v2(rho, v_lim)

#finite difference method
def finite_difference(rho_init, dt, dx, time_steps, flux_func):
    rho = rho_init.copy()
    history = []

    for n in range(time_steps):
        f = flux_func(rho)  # flux array
        rho_next = np.zeros_like(rho)
        
#finite difference method applied
        rho_next[1:-1] = 0.5*(rho[:-2] + rho[2:]) \
                         - (dt/(2*dx))*(f[2:] - f[:-2])
        
        # boundary condition
        rho_next[0] = rho_next[1]
        rho_next[-1] = rho_next[-2]
        
        #limite the rho value
        rho_next = np.clip(rho_next, 0, rho_max)

        rho = rho_next
        history.append(rho.copy())
    return np.array(history)

#initial consition setting
rho_init = np.where(x < L / 2, 0.25, 0.0)

v_lim = 100.0


#compute two different v
density_hist_v1 = finite_difference(rho_init, dt, dx, time_steps, 
                                    lambda r: flux1(r, v_lim))
density_hist_v2 = finite_difference(rho_init, dt, dx, time_steps, 
                                    lambda r: flux2(r, v_lim))

#plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

# density(time-space) diagram(v1)
im1 = axes[0,0].imshow(density_hist_v1, aspect='auto', cmap='viridis',
                       origin='lower', extent=[0, L, 0, T])
axes[0,0].set_title("v1: Space-Time Density")
axes[0,0].set_xlabel("Space (m)")
axes[0,0].set_ylabel("Time (s)")
fig.colorbar(im1, ax=axes[0,0], label='Density')

#snapshot(v1)
times_to_plot = [0, int(time_steps/4), int(time_steps/2), time_steps-1]
for t_idx in times_to_plot:
    axes[0,1].plot(x, density_hist_v1[t_idx], label=f"t={t_idx*dt:.1f}s")
axes[0,1].set_title("v1: Density Snapshots")
axes[0,1].set_xlabel("Space (m)")
axes[0,1].set_ylabel("Density")
axes[0,1].legend()
axes[0,1].grid(True)

# density(time-space) diagram(v2)
im2 = axes[1,0].imshow(density_hist_v2, aspect='auto', cmap='viridis',
                       origin='lower', extent=[0, L, 0, T])
axes[1,0].set_title("v2: Space-Time Density")
axes[1,0].set_xlabel("Space (m)")
axes[1,0].set_ylabel("Time (s)")
fig.colorbar(im2, ax=axes[1,0], label='Density')

#snapshot(v2)
for t_idx in times_to_plot:
    axes[1,1].plot(x, density_hist_v2[t_idx], label=f"t={t_idx*dt:.1f}s")
axes[1,1].set_title("v2: Density Snapshots")
axes[1,1].set_xlabel("Space (m)")
axes[1,1].set_ylabel("Density")
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.show()







#flow rate-density plot in PDE
import numpy as np
import matplotlib.pyplot as plt

# Parameter set
L = 1000  # Length of road (m)
N_x = 200  # Number of spatial grid points
T = 60  # Total simulation time (s)
dt = 0.05  # Time step (s)
rho_max = 0.25  # Max density (vehicles/m)
v_max = 30  # Max velocity (m/s)
dx = L / N_x  # Space step (m)
time_steps = int(T / dt)  # Number of time steps
v_lim = 100  # Speed limit

# Flux functions
def flux_1(rho, v_lim):
    return rho * np.minimum(v_max * (1 - rho / rho_max), v_lim)

def flux_2(rho, v_lim):
    return rho * np.minimum(v_max * (1 - (rho / rho_max))**2, v_lim)

# Finite Difference Method (FDM) for density evolution
def finite_difference(rho_init, dt, dx, time_steps, flux_function, v_lim):
    rho = rho_init.copy()  # Initial density
    density_history = []  # Store density at each time step
    flux_history = []  # Store flux at each time step

    for _ in range(time_steps):
        f = flux_function(rho, v_lim)
        rho_next = np.zeros_like(rho)
        rho_next[1:-1] = 0.5 * (rho[:-2] + rho[2:]) - dt / (2 * dx) * (f[2:] - f[:-2])
        rho_next = np.clip(rho_next, 0, rho_max)

        # Neumann boundary condition
        rho_next[0] = rho_next[1]
        rho_next[-1] = rho_next[-2]

        rho = rho_next
        density_history.append(rho.copy())
        flux_history.append(f.copy())

    return np.array(density_history), np.array(flux_history)

# Initial density: Traffic jam at half road (Neumann boundary)
x = np.linspace(0, L, N_x)
rho_init = np.where(x < L / 2, 0.25, 0.0)

# Solve using FDM for both flux functions
density_history_1, flux_history_1 = finite_difference(rho_init, dt, dx, time_steps, flux_1, v_lim)
density_history_2, flux_history_2 = finite_difference(rho_init, dt, dx, time_steps, flux_2, v_lim)

# Flatten all density and corresponding flux values
all_density_1 = density_history_1.flatten()
all_flux_1 = flux_history_1.flatten()
all_density_2 = density_history_2.flatten()
all_flux_2 = flux_history_2.flatten()

# Select a small subset of points for plotting
sample_indices = np.random.choice(len(all_density_1), size=len(all_density_1)//100, replace=False)
sampled_density_1 = all_density_1[sample_indices]
sampled_flux_1 = all_flux_1[sample_indices]
sampled_density_2 = all_density_2[sample_indices]
sampled_flux_2 = all_flux_2[sample_indices]

# Plot the flow-density relationships
plt.figure(figsize=(10, 6))
plt.scatter(sampled_density_1, sampled_flux_1, s=10, alpha=0.5, color='blue', label=r'Flux vs Density: $v = v_{max} (1 - \rho / \rho_{max})$')
plt.scatter(sampled_density_2, sampled_flux_2, s=10, alpha=0.5, color='red', label=r'Flux vs Density: $v = v_{max} (1 - (\rho / \rho_{max})^2)$')
plt.xlabel('Density (vehicles per meter)')
plt.ylabel('Flow (vehicles per second)')
plt.title('Flow-Density Relationship')
plt.legend()
plt.grid(True)
plt.show()

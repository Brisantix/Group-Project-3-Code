#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 21:18:46 2025

@author: menglingshu
"""





#generic algorithm to compute parameter in GM model
import numpy as np
import pandas as pd
import random
from scipy.interpolate import interp1d
from deap import base, creator, tools, algorithms

# read data
df = pd.read_csv("Car.csv")
# #for car 2
# vehicle_ids = [1, 11] 
##for car 3
vehicle_ids = [11, 24]  
df.iloc[:, 3] = (df.iloc[:, 3] - 1113433200000) / 1000  

filtered_df = df[
    (df.iloc[:, 0].isin(vehicle_ids)) &
    (df.iloc[:, 3] > 0) &
    (df.iloc[:, 3] < (1113433224200 - 1113433200000) / 1000)
].sort_values(by=df.columns[3])

# initial states
initial_speeds = filtered_df.iloc[:, 11].values * 0.3048  
initial_positions = [0, -10]

#leading car data
# #for car 2
#leader_data = filtered_df[filtered_df.iloc[:, 0] == 1]
leader_data = filtered_df[filtered_df.iloc[:, 0] == 11]
leader_times = leader_data.iloc[:, 3].values  
leader_speeds = leader_data.iloc[:, 11].values * 0.3048  

# interpolate leading car data
dt = 0.2  
sim_times = np.arange(0, 24.2, dt)

leader_speed_interp = interp1d(leader_times, leader_speeds, kind='linear', fill_value="extrapolate")
leader_speeds_sim = leader_speed_interp(sim_times)

# compute GM model mse
def evaluate(params):
    alpha, m, l = params

    positions_gm = np.array(initial_positions)
    speeds_gm = np.array([leader_speeds_sim[0], leader_speeds_sim[0]])
    gaps_history_gm = []

    for t in range(len(sim_times) - 1):
        gap = max(positions_gm[0] - positions_gm[1], 0.1)  
        gaps_history_gm.append(gap)
        speeds_gm[0] = leader_speeds_sim[t]  

        new_acc = np.clip(alpha * (speeds_gm[1]**m / (gap**l)) * (speeds_gm[0] - speeds_gm[1]), -10, 10)

        speeds_gm[1] += new_acc * dt
        speeds_gm[1] = max(0, speeds_gm[1])  
        positions_gm[1] += speeds_gm[1] * dt
        positions_gm[0] += speeds_gm[0] * dt

    gaps_history_gm = np.array(gaps_history_gm)
    
    mse_speed_gm = np.mean((leader_speeds_sim - speeds_gm[1])**2)
    mse_headway_gm = np.mean((gaps_history_gm - 10)**2)  
    
    return (mse_speed_gm + mse_headway_gm,)

# generic algorithm computing
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

#range for parameter
toolbox.register("attr_alpha", random.uniform, 0.17, 0.74)  
toolbox.register("attr_m", random.uniform, -2.0, 2.0)  
toolbox.register("attr_l", random.uniform, -1.0, 4.0)  

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_alpha, toolbox.attr_m, toolbox.attr_l), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Crossover, variation, selection
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# generic algorithm computing
population = toolbox.population(n=50)  
NGEN = 40  
CXPB = 0.5  
MUTPB = 0.2  

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = list(map(toolbox.evaluate, offspring))
    
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

best_individual = tools.selBest(population, k=1)[0]
print(f"Optimized GM Model Parameters: alpha={best_individual[0]:.4f}, "
      f"m={best_individual[1]:.4f}, l={best_individual[2]:.4f}")

#compute optimized mse
mse = evaluate(best_individual)
print(f"Optimized GM Model MSE: {mse[0]:.6f}")


# Generic algorithm to optimize paramter
import numpy as np
import pandas as pd
import random
from scipy.interpolate import interp1d
from deap import base, creator, tools, algorithms

#read and process data
df = pd.read_csv("Car.csv")
# #for car 2
# vehicle_ids = [1, 11] 
##for car 3
vehicle_ids = [11, 24]  
df.iloc[:, 3] = (df.iloc[:, 3] - 1113433200000) / 1000  

filtered_df = df[
    (df.iloc[:, 0].isin(vehicle_ids)) &
    (df.iloc[:, 3] > 0) &
    (df.iloc[:, 3] < (1113433224200 - 1113433200000) / 1000)
].sort_values(by=df.columns[3])

#initial states
initial_states = filtered_df.loc[filtered_df.groupby(df.columns[0])[df.columns[3]].idxmin()]
initial_speeds = initial_states.iloc[:, 11].values * 0.3048  
initial_headways = initial_states.iloc[:, 22].values * 0.3048  

#initial position
initial_positions = [0, -initial_headways[1]]

#leading car data
# #for car 2
#leader_data = filtered_df[filtered_df.iloc[:, 0] == 1]
leader_data = filtered_df[filtered_df.iloc[:, 0] == 11]#for car 3
leader_times = leader_data.iloc[:, 3].values  
leader_speeds = leader_data.iloc[:, 11].values * 0.3048  

#interpolate for leading car speed
dt = 0.2  
time_steps = int(24.2 / dt)  
sim_times = np.arange(0, time_steps * dt, dt)

leader_speed_interp = interp1d(leader_times, leader_speeds, kind='linear', fill_value="extrapolate")
leader_speeds_sim = leader_speed_interp(sim_times)

# compute mse in gipp's model
def evaluate(params):
    v_d, a_n, d_n, S_n1, d_prime_n1 = params

    positions_gipps = np.array(initial_positions)
    speeds_gipps = initial_speeds.copy()
    gaps_history_gipps = []

    v = speeds_gipps.copy()
    x = positions_gipps.copy()

    for t in range(time_steps):
        v_next = np.zeros(2)
        v_next[0] = leader_speeds_sim[t]  

        i = 1
        v_acc = v[i] + 2.5 * a_n * dt * (1 - v[i] / v_d) * np.sqrt(0.025 + v[i] / v_d)
        d_safe = x[0] - x[i] - S_n1
        term = (d_n * dt)**2 + d_n * (2 * d_safe - v[i] * dt + (v[0]**2 / d_prime_n1))
        
        if term >= 0:
            v_dec = -d_n * dt + np.sqrt(term)
        else:
            v_dec = 0
        
        v_next[i] = min(v_acc, v_dec)
        v_next[i] = max(0, v_next[i])  

        v = v_next
        x = x + v * dt
        gaps_history_gipps.append(x[0] - x[i])

    gaps_history_gipps = np.array(gaps_history_gipps)
    
    mse_speed_gipps = np.mean((real_speeds_interp - v[1])**2)
    mse_headway_gipps = np.mean((real_headways_interp - gaps_history_gipps)**2)
    
    return (mse_speed_gipps + mse_headway_gipps,)

# gegeric algorithm computing
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

#range for parameter
toolbox.register("attr_vd", random.uniform, 5, 24)
toolbox.register("attr_an", random.uniform, 3, 11) 
toolbox.register("attr_dn", random.uniform, 1, 11)  
toolbox.register("attr_sn1", random.uniform, 5, 15)
toolbox.register("attr_dprime", random.uniform, 1, 11) 

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_vd, toolbox.attr_an, toolbox.attr_dn, toolbox.attr_sn1, toolbox.attr_dprime), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Crossover, variation, selection
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

#Generic algorithm
population = toolbox.population(n=50)  
NGEN = 40  #iterate for 40 times
CXPB = 0.3  #crossover probability
MUTPB = 0.1  # mutation probability


for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = list(map(toolbox.evaluate, offspring))
    
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

best_individual = tools.selBest(population, k=1)[0]
print(f"Optimized Gipps' Model Parameters: vd={best_individual[0]:.4f}, "
      f"a_n={best_individual[1]:.4f}, d_n={best_individual[2]:.4f}, "
      f"S_n1={best_individual[3]:.4f}, d_prime_n1={best_individual[4]:.4f}")

# compute optimized mse
mse = evaluate(best_individual)
print(f"Optimized Gipps' Model MSE: {mse[0]:.6f}")
print("This is a random method, so every running will have different results.")





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

#Upload and processing the data
df = pd.read_csv("Car.csv")
vehicle_ids = [1, 11]  
df.iloc[:, 3] = (df.iloc[:, 3] - 1113433200000) / 1000  # Transform the time into second
filtered_df = df[
    (df.iloc[:, 0].isin(vehicle_ids)) &
    (df.iloc[:, 3] > 0) &
    (df.iloc[:, 3] < (1113433224200 - 1113433200000) / 1000)
].sort_values(by=df.columns[3])

# initial statess
initial_states = filtered_df.loc[filtered_df.groupby(df.columns[0])[df.columns[3]].idxmin()]
initial_speeds_fps = initial_states.iloc[:, 11].values  # feet/s
initial_headways_ft = initial_states.iloc[:, 22].values  # feet
initial_speeds = initial_speeds_fps * 0.3048  # -->m/s
initial_headways = initial_headways_ft * 0.3048  # -->m
#Initial position
initial_positions = [0, -initial_headways[1]]

# leading car data
leader_data = filtered_df[filtered_df.iloc[:, 0] == 1]
leader_times = leader_data.iloc[:, 3].values  
leader_speeds = leader_data.iloc[:, 11].values * 0.3048  # m/s

#Simulated parameter
dt = 0.2  #timestep(s)
tau = 0.2  #Reaction time(second)
k = int(tau / dt)  #delay stap
time_steps = int(24.2 / dt) #Number of time step
v_max = 50  #maximum velocity

#GM model parameter
alpha = 1.7389  #sensitivity parameter
m = 1.7012  
l = 1.8485  



# Gipps' model parameter
vd = 11.4393  # expected velocity
a_n = 5.0987  # maximum acceleration
d_n = 3.3182  # maximum deceleration
d_prime_n1 = 11.1998  # leading meaximum deceleration
S_n1 = 11.6354  # minimum safety headway


# interpolating leading car velocity
sim_times = np.arange(0, time_steps * dt, dt)
leader_speed_interp = interp1d(leader_times, leader_speeds, kind='linear', fill_value="extrapolate")
leader_speeds_sim = leader_speed_interp(sim_times)

# Simulation of GM model
positions_gm = np.array(initial_positions)
speeds_gm = initial_speeds.copy()
positions_history_gm = [positions_gm.copy()]
speeds_history_gm = [speeds_gm.copy()]
gaps_history_gm = []

for t in range(time_steps):
    gap = positions_gm[0] - positions_gm[1]
    gaps_history_gm.append(gap)
    
    speeds_gm[0] = leader_speeds_sim[t]  #velocity of leading car
    

    if t < k:
        gap = positions_gm[0] - positions_gm[1]
        v_i = speeds_gm[1]
        v_prev = speeds_gm[0]
    else:
        pos_delayed = positions_history_gm[t - k]
        speeds_delayed = speeds_history_gm[t - k]
        gap = pos_delayed[0] - pos_delayed[1]
        v_i = speeds_delayed[1]
        v_prev = speeds_delayed[0]
    if gap < 0.1:  #collision threshold
        new_acc = -v_i / dt
    else:
        new_acc = alpha * (v_i**m / (gap**l)) * (v_prev - v_i)
    
    speeds_gm[1] += new_acc * dt
    speeds_gm[1] = np.clip(speeds_gm[1], 0, v_max)
    positions_gm[1] += speeds_gm[1] * dt
    positions_gm[0] += speeds_gm[0] * dt
    
    positions_history_gm.append(positions_gm.copy())
    speeds_history_gm.append(speeds_gm.copy())

positions_history_gm = np.array(positions_history_gm[:-1])
speeds_history_gm = np.array(speeds_history_gm[:-1])
gaps_history_gm = np.array(gaps_history_gm)

#Simulation of Gipp's model
positions_gipps = np.array(initial_positions)
speeds_gipps = initial_speeds.copy()
positions_history_gipps = [positions_gipps.copy()]
speeds_history_gipps = [speeds_gipps.copy()]
gaps_history_gipps = []

v = speeds_gipps.copy()
x = positions_gipps.copy()
for t in range(time_steps):
    v_next = np.zeros(2)
    v_next[0] = leader_speeds_sim[t]  # leading car velocity
    
    # compute car11 velocity
    i = 1
    v_acc = v[i] + 2.5 * a_n * dt * (1 - v[i] / vd) * np.sqrt(0.025 + v[i] / vd)
    d_safe = x[0] - x[i] - S_n1

    term = (d_n * dt)**2 + d_n * (2 * d_safe - v[i] * dt + (v[0]**2 / d_prime_n1))
    if term >= 0:
        v_dec = d_n * tau + np.sqrt(term)
    else:
        v_dec = 0
    v_next[i] = min(v_acc, v_dec, v_max)
    
    v = v_next
    x = x + v * dt
    gaps_history_gipps.append(x[0] - x[i])
    positions_history_gipps.append(x.copy())
    speeds_history_gipps.append(v.copy())

positions_history_gipps = np.array(positions_history_gipps[:-1])
speeds_history_gipps = np.array(speeds_history_gipps[:-1])
gaps_history_gipps = np.array(gaps_history_gipps)

#interpolating for real values
real_speeds = {}
real_headways = {}
real_times = {}
for vid in vehicle_ids:
    vid_data = filtered_df[filtered_df.iloc[:, 0] == vid]
    real_times[vid] = vid_data.iloc[:, 3].values
    real_speeds[vid] = vid_data.iloc[:, 11].values * 0.3048  # m/s
    real_headways[vid] = vid_data.iloc[:, 22].values * 0.3048  # m

real_speeds_interp = interp1d(real_times[11], real_speeds[11], kind='linear', fill_value="extrapolate")(sim_times)
real_headways_interp = interp1d(real_times[11], real_headways[11], kind='linear', fill_value="extrapolate")(sim_times)

#plotting
fig, axs = plt.subplots(figsize=(8, 8))

#compare speed
axs.plot(sim_times, real_speeds_interp, label='Real Speed (Car2)', color='black')
axs.plot(sim_times, speeds_history_gm[:, 1], label='GM Model Speed (Car2)', linestyle='--', color='blue')
axs.plot(sim_times, speeds_history_gipps[:, 1], label="Gipps' Model Speed (Car2)", linestyle='-.', color='red')
axs.set_title("Speed Comparison for Car2",fontsize=16)
axs.set_xlabel("Time (s)",fontsize=16)
axs.set_ylabel("Velocity (m/s)",fontsize=16)
axs.legend(prop={'size':12})
plt.savefig("plot2.png") 
axs.grid()


plt.tight_layout()
plt.show()

#compute mse
mse_speed_gm = np.mean((real_speeds_interp - speeds_history_gm[:, 1])**2)
mse_speed_gipps = np.mean((real_speeds_interp - speeds_history_gipps[:, 1])**2)
mse_headway_gm = np.mean((real_headways_interp - gaps_history_gm)**2)
mse_headway_gipps = np.mean((real_headways_interp - gaps_history_gipps)**2)

print(f"GM  - velocity MSE: {mse_speed_gm:.4f}, headway MSE: {mse_headway_gm:.4f}")
print(f"Gipps'  - velocity MSE: {mse_speed_gipps:.4f}, headway MSE: {mse_headway_gipps:.4f}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

#upload and process data
df = pd.read_csv("Car.csv")
vehicle_ids = [11, 24]  \
df.iloc[:, 3] = (df.iloc[:, 3] - 1113433200000) / 1000  # -->second
filtered_df = df[
    (df.iloc[:, 0].isin(vehicle_ids)) &
    (df.iloc[:, 3] > 0) &
    (df.iloc[:, 3] < (1113433224200 - 1113433200000) / 1000)
].sort_values(by=df.columns[3])

#initial states
initial_states = filtered_df.loc[filtered_df.groupby(df.columns[0])[df.columns[3]].idxmin()]
initial_speeds_fps = initial_states.iloc[:, 11].values  # feet/s
initial_headways_ft = initial_states.iloc[:, 22].values  # feet
initial_speeds = initial_speeds_fps * 0.3048  
initial_headways = initial_headways_ft * 0.3048  

# initial position
initial_positions = [0, -initial_headways[1]]

# leading car data
leader_data = filtered_df[filtered_df.iloc[:, 0] == 11]
leader_times = leader_data.iloc[:, 3].values  
leader_speeds = leader_data.iloc[:, 11].values * 0.3048  # m/s

#simulation parameter
dt = 0.2  #timestep(s)
tau = 0.2  #reaction time(s)
k = int(tau / dt)  #delay step
time_steps = int(24.2 / dt) 
v_max = 50  #max speed

#GM model parameter
alpha=0.3952
m=1.9423
l=0.9379


#Gipp's model parameter
vd = 11.2780  #expected velocity
a_n = 3.6835  # max acceleration
d_n = 2.9298  # max deceleration
d_prime_n1 = 3.3182  #leading car max deceleration
S_n1 = 4.5691  # effective vehicle parameter


# interpolating leading car velocity
sim_times = np.arange(0, time_steps * dt, dt)
leader_speed_interp = interp1d(leader_times, leader_speeds, kind='linear', fill_value="extrapolate")
leader_speeds_sim = leader_speed_interp(sim_times)

#simulation of GM model
positions_gm = np.array(initial_positions)
speeds_gm = initial_speeds.copy()
positions_history_gm = [positions_gm.copy()]
speeds_history_gm = [speeds_gm.copy()]
gaps_history_gm = []

for t in range(time_steps):
    gap = positions_gm[0] - positions_gm[1]
    gaps_history_gm.append(gap)
    
    speeds_gm[0] = leader_speeds_sim[t]  # real speed of leading car
    
    #compute acceleration
    if t < k:
        gap = positions_gm[0] - positions_gm[1]
        v_i = speeds_gm[1]
        v_prev = speeds_gm[0]
    else:
        pos_delayed = positions_history_gm[t - k]
        speeds_delayed = speeds_history_gm[t - k]
        gap = pos_delayed[0] - pos_delayed[1]
        v_i = speeds_delayed[1]
        v_prev = speeds_delayed[0]
    if gap < 0.1:  
        new_acc = -v_i / dt
    else:
        new_acc = alpha * (v_i**m / (gap**l)) * (v_prev - v_i)
    
    speeds_gm[1] += new_acc * dt
    speeds_gm[1] = np.clip(speeds_gm[1], 0, v_max)
    positions_gm[1] += speeds_gm[1] * dt
    positions_gm[0] += speeds_gm[0] * dt
    
    positions_history_gm.append(positions_gm.copy())
    speeds_history_gm.append(speeds_gm.copy())

positions_history_gm = np.array(positions_history_gm[:-1])
speeds_history_gm = np.array(speeds_history_gm[:-1])
gaps_history_gm = np.array(gaps_history_gm)

#Gipp model simulation
positions_gipps = np.array(initial_positions)
speeds_gipps = initial_speeds.copy()
positions_history_gipps = [positions_gipps.copy()]
speeds_history_gipps = [speeds_gipps.copy()]
gaps_history_gipps = []

v = speeds_gipps.copy()
x = positions_gipps.copy()
for t in range(time_steps):
    v_next = np.zeros(2)
    v_next[0] = leader_speeds_sim[t]  #leading car real speed
    
    # compute vehicle 11 speed
    i = 1
    v_acc = v[i] + 2.5 * a_n * dt * (1 - v[i] / vd) * np.sqrt(0.025 + v[i] / vd)
    d_safe = x[0] - x[i] - S_n1

    term = (d_n * dt)**2 + d_n * (2 * d_safe - v[i] * dt + (v[0]**2 / d_prime_n1))
    if term >= 0:
        v_dec = d_n * tau + np.sqrt(term)
    else:
        v_dec = 0
    v_next[i] = min(v_acc, v_dec, v_max)
    
    v = v_next
    x = x + v * dt
    gaps_history_gipps.append(x[0] - x[i])
    positions_history_gipps.append(x.copy())
    speeds_history_gipps.append(v.copy())

positions_history_gipps = np.array(positions_history_gipps[:-1])
speeds_history_gipps = np.array(speeds_history_gipps[:-1])
gaps_history_gipps = np.array(gaps_history_gipps)

# interpolate real speed
real_speeds = {}
real_headways = {}
real_times = {}
for vid in vehicle_ids:
    vid_data = filtered_df[filtered_df.iloc[:, 0] == 24]
    real_times[vid] = vid_data.iloc[:, 3].values
    real_speeds[vid] = vid_data.iloc[:, 11].values * 0.3048  # m/s
    real_headways[vid] = vid_data.iloc[:, 22].values * 0.3048  # m

real_speeds_interp = interp1d(real_times[11], real_speeds[11], kind='linear', fill_value="extrapolate")(sim_times)
real_headways_interp = interp1d(real_times[11], real_headways[11], kind='linear', fill_value="extrapolate")(sim_times)

#plotting
fig, axs = plt.subplots(figsize=(8,8))

#compare velocity
axs.plot(sim_times, real_speeds_interp, label='Real Speed (Car3)', color='black')
axs.plot(sim_times, speeds_history_gm[:, 1], label='GM Model Speed (Car3)', linestyle='--', color='blue')
axs.plot(sim_times, speeds_history_gipps[:, 1], label="Gipps' Model Speed (Car3)", linestyle='-.', color='red')
axs.set_title("Speed Comparison for Car3",fontsize=16)
axs.set_xlabel("Time (s)",fontsize=16)
axs.set_ylabel("Velocity (m/s)",fontsize=16)
axs.legend(prop={'size':12})
axs.grid()
plt.savefig("plot2.png") 



plt.tight_layout()
plt.show()

# compute mse
mse_speed_gm = np.mean((real_speeds_interp - speeds_history_gm[:, 1])**2)
mse_speed_gipps = np.mean((real_speeds_interp - speeds_history_gipps[:, 1])**2)
mse_headway_gm = np.mean((real_headways_interp - gaps_history_gm)**2)
mse_headway_gipps = np.mean((real_headways_interp - gaps_history_gipps)**2)

print(f"GM  - velocity MSE: {mse_speed_gm:.4f}, headway MSE: {mse_headway_gm:.4f}")
print(f"Gipps'  - velocity MSE: {mse_speed_gipps:.4f}, headway MSE: {mse_headway_gipps:.4f}")


# put figures together
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
fig, axs = plt.subplots(1, 2, figsize=(20, 12))

img1 = Image.open("plot1.png")
img2 = Image.open("plot2.png")

axs[0].imshow(img1)
axs[0].axis("off")  
axs[0].set_title("First Plot")

axs[1].imshow(img2)
axs[1].axis("off") 
axs[1].set_title("Second Plot")

plt.tight_layout()
plt.show()

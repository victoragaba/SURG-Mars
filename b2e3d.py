import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import function_repo as fr
from collections import defaultdict
import importlib
np.random.seed(1029)



"""
(Normal) fault, plot and save amplitude space
"""
# # For every azimuth-takeoff angle pair, plot the amplitudes

# importlib.reload(fr)
# dd = 3 # step size in degrees, subject to change
# # azimuths = np.arange(0, 360, dd)
# takeoff_angles = np.arange(0, 180, dd)
# alpha_beta = [5.8000, 3.4600] # from Suzan's notes

# # Fault orientations
# t, p = fr.i_hat, fr.k_hat # subject to change
# # Tested my_tp2sdr against tp2sdr, it works properly now
# sdr1, sdr2 = fr.my_tp2sdr(t, p, True)

# # # Plot corresponding beachball
# # fr.beachball(sdr2); 

# # Generate amplitude space
# # Assumes same takeoff angles for P and S waves
# # Typically should come from velocity model
# amps = defaultdict(list)

# # Applying logic of spherical distribution
# for takeoff in takeoff_angles:
#     if takeoff == 0: azimuths = np.array([0])
#     else: azimuths = np.arange(0, 360, dd/abs(np.sin(np.deg2rad(takeoff))))
#     for az in azimuths:
#         As = fr.Rpattern(sdr1, az, [takeoff, takeoff], alpha_beta)
#         amps["AP"].append(As[0])
#         amps["ASV"].append(As[1])
#         amps["ASH"].append(As[2])
#         amps["Standard"].append(0.5)

# # # Plot and save amplitude space
# # fr.weighted_3D_scatter(amps, "Standard", type="amp")
# # Got lucky not to need a dataframe




"""
Realistic amplitude space, using velocity model
"""

# model = TauPyModel(model='ak135') # velocity model
# hdepth = 15 # km - assumed quake depth, subject to change
# epdists = [i for i in range(100)] # km - epicentral distance
# # same azimuths

# realistic_takeoff_angles = []
# rejected_epdists = []
# for epdist in epdists:
#     arrivals = model.get_travel_times(source_depth_in_km=hdepth,
#                          distance_in_degree=epdist, phase_list=['P', 'S'])
#     angles = [a.takeoff_angle for a in arrivals]
#     if len(angles) == 2:
#         realistic_takeoff_angles.append(([a.takeoff_angle for a in arrivals], epdist))
#     else:
#         rejected_epdists.append(epdist)

# print(f"{len(rejected_epdists)}/{len(epdists)} rejected epicentral distances")
# print(rejected_epdists[:5], rejected_epdists[-5:])
        
# # plot realistic amplitude space over original
# # assumption that P and S waves have same takeoff angles no longer holds
# # same amps dictionary
# # extra one to isolate realistic amps
# real_amps = defaultdict(list)

# for takeoff in realistic_takeoff_angles:
#     average_takeoff = np.mean(takeoff[0])
#     azimuths = np.arange(0, 360, dd/abs(np.sin(np.deg2rad(average_takeoff))))
#     for az in azimuths:
#         As = fr.Rpattern(sdr1, az, takeoff[0], alpha_beta)
#         amps["AP"].append(As[0])
#         amps["ASV"].append(As[1])
#         amps["ASH"].append(As[2])
#         amps["Standard"].append(1) # or takeoff[1] for distance gradient
#         real_amps["AP"].append(As[0])
#         real_amps["ASV"].append(As[1])
#         real_amps["ASH"].append(As[2])
#         real_amps["Standard"].append(takeoff[1])
        
# # Plot and save amplitude space
# fr.weighted_3D_scatter(amps, "Standard", type="amp", save="real_amp_space.png")
# fr.weighted_3D_scatter(real_amps, "Standard", type="amp", save="real_amps.png")

# "See also real_amps_with_grad.png"
# # Do the plots myself, they're too variable to automate



"""
sdr grid search
"""

# Remember to account for sinusoidal distribution of dips - inverse transform sampling
# This may solve my geometry problem from tp -> sdr conversion

d_dip = 5 # step size in degrees, subject to change
dips = np.arange(0, 90, d_dip) # uniform samples from distribution of interest
sdrs = []

for dip in dips[1:]: # 0 dip makes no sense
    d_adj = d_dip/np.sin(np.deg2rad(dip))
    strikes = np.arange(0, 360, d_adj) # strikes restricted by dip
    rakes = np.arange(-180, 180, d_dip) # rakes dependent on sd pair
    for strike in strikes:
        for rake in rakes:
            sdrs.append(np.array([strike, dip, rake]))

# # save sdrs
# np.save("sdrs.npy", sdrs)

# Forward problem from sdr space for a fixed az and takeoff

az = 0
model = TauPyModel(model='ak135')
hdepth = 15
epdist = 45
arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                         distance_in_degree=epdist, phase_list=['P', 'S'])
angles = [a.takeoff_angle for a in arrivals]
sdr_amps = defaultdict(list)

alpha_beta = [5.8000, 3.4600]
for sdr in sdrs:
    As = fr.Rpattern(sdr, az, angles, alpha_beta)
    sdr_amps["AP"].append(As[0])
    sdr_amps["ASV"].append(As[1])
    sdr_amps["ASH"].append(As[2])
    sdr_amps["Standard"].append(1)
    
# Plot and save amplitude space
# fr.weighted_3D_scatter(sdr_amps, "Standard", type="amp", save="sdr_amp_space.png")
fig = plt.figure(figsize=(15,12))
ax = plt.axes(projection='3d')
scatter = ax.scatter3D(sdr_amps["AP"], sdr_amps["ASV"], sdr_amps["ASH"], s=1/10)
ax.set_xlabel('AP')
ax.set_ylabel('ASV')
ax.set_zlabel('ASH')
plt.show()
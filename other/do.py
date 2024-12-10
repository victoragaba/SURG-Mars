import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import function_repo as fr
from collections import defaultdict
import importlib
np.random.seed(1029)

# load dataframe
sdr_amps = pd.read_csv("sdr_amps.csv")

# Plot and save amplitude space
# fr.weighted_3D_scatter(sdr_amps, "Standard", type="amp", save="sdr_amp_space.png")
fig = plt.figure(figsize=(15,12))
ax = plt.axes(projection='3d')
scatter = ax.scatter3D(sdr_amps["AP"], sdr_amps["ASV"], sdr_amps["ASH"], s=1/20)
# set title
ax.set_title("Amplitude Space from T-P Axes", fontsize=20)
# make axes equal
ax.set_aspect('equal')
ax.set_xlabel('AP')
ax.set_ylabel('ASV')
ax.set_zlabel('ASH')
plt.show()

"Conclusion" # I need random sampling (inverse transform sampling)
# To see if the patterns in the middle are real or just a result of the grid search
# I hope it's the latter, because there's no reason (yet) for the middle to be special
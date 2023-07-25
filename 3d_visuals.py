import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import function_repo as fr
import importlib
importlib.reload(fr)

df = pd.read_csv("normal_fault_guess.csv")

old_accepted1 = df["OldWeight"] >= np.exp(-1/2)
old_accepted2 = (df["OldWeight"] >= np.exp(-2)) & (df["OldWeight"] < np.exp(-1/2))
old_rejected = df["OldWeight"] < np.exp(-2)
accepted1 = df["Weight"] >= np.exp(-1/2)
accepted2 = (df["Weight"] >= np.exp(-2)) & (df["Weight"] < np.exp(-1/2))
rejected = df["Weight"] < np.exp(-2)

t, p = np.array([1,0,0]), np.array([0,0,1])
normal_faults = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))
normal_faults = np.rad2deg(np.array(normal_faults))

fr.weighted_3D_scatter(df[old_accepted1], "OldWeight", normal_faults)
fr.weighted_3D_scatter(df[accepted1], "Weight", normal_faults)

# Continue from here... red star
# fr.weighted_3D_scatter(df, "Weight")
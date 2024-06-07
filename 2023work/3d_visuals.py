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

# t, p = np.array([1,0,0]), np.array([0,0,1])
# normal_faults = fr.tp2sdr(fr.coord_switch(t), fr.coord_switch(p))
# normal_faults = np.rad2deg(np.array(normal_faults))

# # fr.weighted_3D_scatter(df[old_accepted1], "OldWeight", normal_faults)
# # fr.weighted_3D_scatter(df[accepted1], "Weight", normal_faults)

# fr.weighted_3D_scatter(df[old_accepted1], "OldWeight", type="tpa")
# fr.weighted_3D_scatter(df[accepted1], "Weight", type="tpa")

# X,Y,Z = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10), np.linspace(-1,1,10))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X,Y,Z)
# plt.show()

# X,Y,Z = np.linspace(-1,1,10), np.linspace(-1,1,10), np.linspace(-1,1,10)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X,Y,Z)
# plt.show()

# X,Y,Z = np.linspace(-1,1,10), np.linspace(-1,1,10), np.linspace(-1,1,10)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X,Y,Z)
# plt.show()

# # plot a 3d line segment
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # extend axes
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_zlim(-2,2)
# t = np.array([1,1,1])/np.sqrt(3)
# # X,Y,Z = [-1,1], [-1,1], [-1,1]
# # X2,Y2,Z2 = [1,-1], [-1,1], [-1,1]
# ## overlay the line segments
# # ax.plot(X,Y, Z)
# # ax.plot(X2,Y2,Z2)
# fr.plot_cross(ax, t, fr.starting_direc(t,fr.j_hat))
# plt.show()




fr.plot_crosses(df[old_accepted1], "OldWeight")
fr.plot_crosses(df[accepted1], "Weight")




# x = fr.beachball([45., 45., -90.], alpha=0.5, edgecolor=None)
# print(type(x))
# x.figimage((1,2))


# fig, ax = plt.subplots()
# collection = fr.beach([45., 45., -90.], alpha=0.5, edgecolor=None)
# collection2 = fr.beach([0, 45., -180.], alpha=0.5, edgecolor=None)
# print(type(collection))
# ax.add_collection(collection)
# ax.add_collection(collection2)
# ax.set_xlim(-100,100)
# ax.set_ylim(-100,100)
# plt.show()




# print(len(df[old_accepted1]))
# fr.plot_beachball_set(df[old_accepted1], "OldWeight")
# fr.plot_beachball_set(df[accepted1], "Weight")




# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from matplotlib.collections import PatchCollection

# # Create a new figure and subplot
# fig, ax = plt.subplots()

# # Example data: centers and radii of circles
# centers = [(1, 1), (2, 3), (4, 2)]
# radii = [0.5, 0.3, 0.7]

# # Create a list of Circle patches
# circle_patches = [Circle(center, radius) for center, radius in zip(centers, radii)]

# # Create a PatchCollection from the Circle patches
# circle_collection = PatchCollection(circle_patches, edgecolor='blue', facecolor='none')

# # Add the PatchCollection to the subplot
# ax.add_collection(circle_collection)

# # Set limits and labels
# # ax.set_xlim(0, 5)
# # ax.set_ylim(0, 4)
# # ax.set_xlabel('X-axis')
# # ax.set_ylabel('Y-axis')
# # ax.set_title('Circle Collection')

# # Show the plot
# # plt.show()

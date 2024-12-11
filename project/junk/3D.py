import numpy as np
from numpy import linalg
import functions as fn
import seismic_model as sm
import optimizer as opt
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
import importlib

np.random.seed(2024)
# np.random.seed(np.random.randint(0, 2024))

# create random array for t, normalised to 1
t = fn.unit_vec(np.random.rand(3))

# create random array for p, normalised to 1
direc = fn.unit_vec(np.random.rand(3))
p = fn.starting_direc(t, direc)

# get truen params for synthetic test
true_params = fn.tp2sdr(t, p, True)[0]
print('Original params: ', true_params)

importlib.reload(fn)
importlib.reload(sm)
importlib.reload(opt)

# set up parameters for inversion
model = TauPyModel(model='ak135')  # velocity model
hdepth = 15  # km - assumed quake depth
epdist = 10  # degrees - epicentral distance
azimuth = 200  # degrees - azimuth of seismometer
arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                        distance_in_degree=epdist, phase_list=['P', 'S'])
takeoff_angles = [a.takeoff_angle for a in arrivals]
velocities = np.array([5.8000, 3.4600])

# initialize model
inv_model = sm.SeismicModel(np.deg2rad(azimuth), takeoff_angles, velocities)
best_fit = inv_model(np.deg2rad(true_params), set_Ao=True)



######################


Ao = inv_model.get_Ao()
inv_model.set_Uo()
Uo = inv_model.get_Uo()
chi2 = inv_model.get_chi2()
normal = inv_model.get_normal()
dd = 15
n = 50 # enough to cover a half-revolution
amps = fn.systematic_elliptical_cone(normal, Ao, Uo, chi2, dd, n)

# plot amps in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot([0, normal[0]], [0, normal[1]], [0, normal[2]], c='b')
ax.scatter(amps[0], amps[1], amps[2], c='r', marker='o')
ax.set_xlabel('AP')
ax.set_ylabel('ASV')
ax.set_zlabel('ASH')
ax.set_title('Simulated amplitude data')
plt.show()

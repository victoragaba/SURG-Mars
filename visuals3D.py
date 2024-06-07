from numpy import *
import matplotlib.pyplot as plt
import seismic_model as sm
import minimize_objective as mo
from obspy.taup import TauPyModel
import functions as fx
import importlib

eps = 1e-10; halfpi = pi/2; twopi = 2*pi
i_hat = array([1,0,0]); j_hat = array([0,1,0]); k_hat = array([0,0,1])
random.seed(1234)





# create random array for t, normalised to 1
t = random.rand(3)
t /= linalg.norm(t)

# create random array for p, normalised to 1
normal = random.rand(3)
normal /= linalg.norm(normal)
p = fx.starting_direc(t, normal)

# get sdr for test case
sdr, sdr_alt = fx.tp2sdr(t, p, True)
print(f'Original sdr: {sdr}')

# plot corresponding beachball
# beachball = fx.beachball(sdr)





# set up parameters for inversion
model = TauPyModel(model='ak135')  # velocity model

'''
THESE SHOULD BE CHANGED LATER
'''
hdepth = 15  # km - assumed quake depth
epdist = 10  # degrees - epicentral distance
azimuth = 200  # degrees - azimuth of seismometer


arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                        distance_in_degree=epdist, phase_list=['P', 'S'])
takeoff_angles = [a.takeoff_angle for a in arrivals]
alpha, beta = 5.8000, 3.4600
b3_over_a3 = (beta/alpha)**3

# generate observed amplitude vector
Ao = array(fx.Rpattern(sdr, azimuth, takeoff_angles, [alpha, beta]))

# beachball
# beachball = fx.beachball(sdr)





#########################################

# MODEL SPACE

# Generate sdr grid
step_size = 20  # in degrees\
s_range = arange(0, 360, step_size)
d_range = arange(0, 90, step_size)
r_range = arange(-180, 180, step_size)

sdr_grid = zeros((len(s_range), len(d_range), len(r_range), 3))
cossims = zeros((len(s_range), len(d_range), len(r_range)))
# sdr_grid.shape

for i in range(len(s_range)):
    for j in range(len(d_range)):
        for k in range(len(r_range)):
            sdr = array([s_range[i], d_range[j], r_range[k]])
            sdr_grid[i, j, k] = sdr
            amplitude = fx.Rpattern(sdr, azimuth, takeoff_angles, [alpha, beta])
            cossims[i, j, k] = fx.cossim(Ao, amplitude)



# names of common cmaps
# 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
# 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',



# plot the cossim grid on sdr scatterplot
fig5 = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))

strkes = sdr_grid[:,:,:,0].flatten()
dips = sdr_grid[:,:,:,1].flatten()
rakes = sdr_grid[:,:,:,2].flatten()
cossims = cossims.flatten()
# get where cossims > threshold
threshold = 0.9
indices = where(cossims > threshold)
high_strkes = strkes[indices]
high_dips = dips[indices]
high_rakes = rakes[indices]

scatter = ax.scatter3D(strkes, dips, rakes, c=cossims, cmap='viridis', s=1)
ax.scatter3D(high_strkes, high_dips, high_rakes, c='r', s=10)
ax.set_title("SDR Samples")
ax.set_xlabel('Strike')
ax.set_ylabel('Dip')
ax.set_zlabel('Rake')
# add high cossim points in red
cbar = fig5.colorbar(scatter, ax=ax, orientation='horizontal')
plt.show()









###########################################


# # construct rigid hemisphere samples to set up source grid
# importlib.reload(fx)

# step_size = 20 # in degrees
# Ts = fx.rigid_hemisphere_samples(step_size, angles=False)
# Ts = array(Ts)

# fig1 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_box_aspect((1,1,.5))
# ax.scatter3D([T[0] for T in Ts],
#                 [T[1] for T in Ts],
#                 [T[2] for T in Ts], c='b', s=1)
# ax.set_title("T Axis Samples")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # plt.show()





# # plot p-axis samples
# Ps = []
# normal = array([0, 0, 1]) # peak of hemisphere
# for T in Ts:
#     P_start = fx.starting_direc(T, normal)
#     for theta in range(0, 180, step_size):
#         P = fx.rotate_vector(P_start, T, deg2rad(theta))
#         Ps.append(P)
# Ps = array(Ps)

# # red color
# fig2 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_box_aspect((1,1,.5))
# ax.scatter3D([P[0] for P in Ps],
#                 [P[1] for P in Ps],
#                 [P[2] for P in Ps], c='r', s=1/2)
# ax.set_title("P Axis Samples")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # plt.show()




# # seasons are always temporary
# # organize yourself to outlast the season
# # do what you say you're going to do
# # it's not over until i win
# # you will grow through what you go through
# # turn the switch on and keep it on
# # outlast the pain
# # there's a joy that overpowers my will
# # i'm not going to let you go until you bless me




# ratio = int(len(Ps)/len(Ts))
# sdrs = []
# sdrs_alt = []
# for i in range(len(Ts)):
#     t = Ts[i]
#     for j in range(ratio):
#         p = Ps[i*ratio + j]
#         sdr, sdr_alt = fx.tp2sdr(t, p, True)
#         sdrs.append(sdr)
#         sdrs_alt.append(sdr_alt)
# sdrs = array(sdrs)
# sdrs_alt = array(sdrs_alt)

# # 3d plot of sdrs/sdrs_alt
# fig3 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_box_aspect((1,1,1))
# ax.scatter3D(sdrs_alt[:,0], sdrs_alt[:,1], sdrs_alt[:,2], c='b', s=1)
# ax.set_title("SDR Samples")
# ax.set_xlabel('Strike')
# ax.set_ylabel('Dip')
# ax.set_zlabel('Rake')
# plt.show()



#############################################
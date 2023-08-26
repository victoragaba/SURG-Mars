from function_repo import *
import numpy as np

"""
MY NOTES:

Random samples over rigid ones (more computationally reliable)

Consider starting at pole for rigid sampling

TP distribution
"""

if __name__ == '__main__':
    
    """
    Random samples over upper hemisphere
    # """
    # N = 10000
    # random_samples = random_hemisphere_samples(N)
    # # t_sphericals = fill_square_randomly(N, [0,0,twopi,halfpi])
    # # random_samples = [pol2rect([1, t[0], t[1]]) for t in t_sphericals]
        
    # fig1 = plt.figure()
    # ax = plt.axes(projection='3d')
    # fig = ax.scatter3D([p[0] for p in random_samples],
    #                 [p[1] for p in random_samples],
    #                 [p[2] for p in random_samples], c='b', s=1000/N)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    
    """
    Rigid samples over upper hemisphere
    """
    # polar_grid = []
    # n = 10 # per grid unit
    # # truncation creates a hole when sin(phi) ~ 0
    # # large n conteracts effect
    # num_divs = 50 # rigid sample
    
    # theta_grid = space_evenly(num_divs, [0, 2*np.pi])
    # div_theta = theta_grid[1] - theta_grid[0]
    # phi_grid = space_evenly(num_divs, [0, np.pi/2])
    # div_phi = phi_grid[1] - phi_grid[0]
    
    # for i in range(len(theta_grid)):
    #     for j in range(len(phi_grid)):
    #         corrected_n = int(n*np.sin(phi_grid[j]))
    #         bounds = [theta_grid[i], phi_grid[j], theta_grid[i] + div_theta, phi_grid[j] + div_phi]
    #         polar_grid.extend(fill_square_randomly(corrected_n, bounds))
            
    # rigid_samples = [pol2rect(np.array([1, p[0], p[1]])) for p in polar_grid]
    
    # fig2 = plt.figure()
    # ax = plt.axes(projection='3d')
    # fig = ax.scatter3D([p[0] for p in rigid_samples],
    #                 [p[1] for p in rigid_samples],
    #                 [p[2] for p in rigid_samples], c='b', s=1/10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    
    
    """
    Rigid samples, reverse order
    """
    # n = 40
    # d_phi = np.deg2rad(1.5)
    # phis = np.arange(0, halfpi, d_phi)
    # real_samples = [k_hat]
    
    # for phi in phis[1:]:
    #     c_phi = twopi*np.sin(phi)
    #     thetas = np.linspace(0, twopi, int(c_phi/d_phi)+1)[:-1]
    #     if len(thetas) == 0: print(f"No points at phi={phi}")
    #     real_samples.extend([pol2rect(np.array([1, theta, phi])) for theta in thetas])
    
    # fig3 = plt.figure()
    # ax = plt.axes(projection='3d')
    # fig = ax.scatter3D([p[0] for p in real_samples],
    #                 [p[1] for p in real_samples],
    #                 [p[2] for p in real_samples], c='b', s=1/10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # print(len(real_samples))
    # plt.show()
    
    
    """
    Suzan's version
    """
    # dd = np.deg2rad(1.5)
    # angles = np.arange(0,np.pi,dd)
    # Ts = [np.array([-1,0,0])] # southpole
    # sphcs = [np.array([0,0])]
    
    # for a in angles[1:]:    # a is "take-off angle" of T axis
    #                     #(measured w.r.t. downpointing vertical) ("latitude")
    #     ddo = dd/np.sin(a)  # need fewer sampling points on small-circle when close to the poles
    #     angleso = np.arange(0,np.pi,ddo)
    #     for o in angleso:   # o is azimuth of T axis ("longitude")
    #         T = np.array([-np.cos(a), -np.sin(a)*np.cos(o), np.sin(a)*np.sin(o)])
    #         Ts.append(T)
    #         sphcs.append(np.degrees(np.array([o,a])))
    
    # fig4 = plt.figure()
    # ax = plt.axes(projection='3d')
    # # fig = ax.scatter3D([T[2] for T in Ts],
    # #                 [T[0] for T in Ts],
    # #                 [-T[1] for T in Ts], c='b', s=1/10)
    # fig = ax.scatter3D([T[0] for T in Ts],
    #                 [T[1] for T in Ts],
    #                 [T[2] for T in Ts], c='b', s=1/10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # print(len(Ts))
    # plt.show()
    
    """
    Plotting p axes
    """
    dd = 5
    Ts = rigid_hemisphere_samples(dd)
    # Ts = Ts[:int(len(Ts)/8)]
    Ts = Ts[::64]
    # Ts = Ts[:16]
    Ps = []
    P_rotations = np.arange(0, np.pi, np.deg2rad(dd))
    test = np.array([1,1,1])
    for T in Ts:
        P_start = starting_direc(T, j_hat)
        Ps.extend([rodrigues_rotate(P_start, T, theta) for theta in P_rotations])
    
    fig5 = plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D([T[0] for T in Ts],
                    [T[1] for T in Ts],
                    [T[2] for T in Ts], c='b', s=1/10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    print(len(Ps))
    # plt.show()
    
    fig6 = plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D([P[0] for P in Ps],
                    [P[1] for P in Ps],
                    [P[2] for P in Ps], c='b', s=1/10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    print(len(Ps))
    plt.show()
    
# Spot checks per t axis, look at half rotation of p's that comes from it
# Look at Suzan's code for t-p axis generation
# Plot the sphere of p axes
# Implement the amplitude space 
# Try reducing the number of p axes per t axis
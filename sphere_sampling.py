from function_repo import *

"""
MY NOTES:

Random samples over rigid ones (more computationally reliable)

Consider starting at pole for rigid sampling

TP distribution
"""

if __name__ == '__main__':
    
    """
    Random samples over upper hemisphere
    """
    N = 10000
    random_samples = hemisphere_samples(N)
    # t_sphericals = fill_square_randomly(N, [0,0,twopi,halfpi])
    # random_samples = [pol2rect([1, t[0], t[1]]) for t in t_sphericals]
        
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D([p[0] for p in random_samples],
                    [p[1] for p in random_samples],
                    [p[2] for p in random_samples], c='b', s=1000/N)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    """
    Rigid samples over upper hemisphere
    """
    polar_grid = []
    n = 10 # per grid unit
    # truncation creates a hole when sin(phi) ~ 0
    # large n conteracts effect
    num_divs = 50 # rigid sample
    
    theta_grid = space_evenly(num_divs, [0, 2*np.pi])
    div_theta = theta_grid[1] - theta_grid[0]
    phi_grid = space_evenly(num_divs, [0, np.pi/2])
    div_phi = phi_grid[1] - phi_grid[0]
    
    for i in range(len(theta_grid)):
        for j in range(len(phi_grid)):
            corrected_n = int(n*np.sin(phi_grid[j]))
            bounds = [theta_grid[i], phi_grid[j], theta_grid[i] + div_theta, phi_grid[j] + div_phi]
            polar_grid.extend(fill_square_randomly(corrected_n, bounds))
            
    rigid_samples = [pol2rect(np.array([1, p[0], p[1]])) for p in polar_grid]
    
    fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D([p[0] for p in rigid_samples],
                    [p[1] for p in rigid_samples],
                    [p[2] for p in rigid_samples], c='b', s=1/10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
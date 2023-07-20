from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

"""
MY NOTES:

Random samples over rigid ones (more computationally reliable)

Consider starting at pole for rigid sampling

TP distribution
"""

eps = 1e-8; halfpi = np.pi/2; twopi = 2*np.pi
i_hat = np.array([1,0,0]); j_hat = np.array([0,1,0]); k_hat = np.array([0,0,1])

def set_axes_equal(ax): # might be useful later
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def pol2rect(pol: list) -> list:
    """
    Convert from polar/spherical to rectangular coordinates

    Args:
        pol (list[int]): polar/spherical coordinates
    """
    r = pol[0]
    theta = pol[1]
    
    if len(pol) == 2:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return np.array([x,y])
    
    phi = pol[2]
    
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    
    return np.array([x,y,z])

def fill_square_randomly(n: int, bounds: list) -> list:
    """
    Fill 2D grid with n uniformly distributed points
    Return array of coordinates w.r.t grid boundaries

    Args:
        n (int): # grid points
        bounds (list[float]): grid bounds [start_hor, start_vert, end_hor, end_vert]
    """
    
    filled = []
    div_hor = bounds[2] - bounds[0]
    div_vert = bounds[3] - bounds[1]
    
    for i in range(n):
        p_hor = stats.uniform.rvs(bounds[0], div_hor)
        p_vert = stats.uniform.rvs(bounds[1], div_vert)
        filled.append(np.array([p_hor, p_vert]))
        
    return filled

def space_evenly(n: int, bounds: list) -> list:
    """
    Return array of n evenly spaced points within specified bounds in 1D

    Args:
        n (int): # divisions
        bounds (list[float]): array bounds [start, end]
    """
    
    spaced = []
    div = (bounds[1] - bounds[0])/n
    for i in range(n):
        spaced.append(bounds[0] + i*div)
    
    return spaced

def hemisphere_samples(n: int) -> list: # important
    """
    Select n random samples on the surface of an upper hemisphere

    Args:
        n (int): # samples

    Returns:
        list: array of [x,y,z] coordinates
    """
    samples = []

    for i in range(n):
        p = stats.norm.rvs(size=3)
        while np.linalg.norm(p) < 0.00001 or all(p[:2]) == 0:
            p = stats.norm.rvs(size=3)
        p /= np.linalg.norm(p)
        p[2] = abs(p[2]) # only upper hemisphere
        samples.append(p)
    
    return samples

if __name__ == '__main__':
    
    """
    Random samples over upper hemisphere
    """
    N = 15000
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
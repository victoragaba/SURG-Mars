import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import collections

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

def hemisphere_samples(n: int) -> list:
    """
    Select n random samples uniformly distributed on the surface of an upper hemisphere
    of unit radius

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
        p[2] = np.abs(p[2]) # only upper hemisphere
        samples.append(p)
    
    return samples

def uniform_samples(n: int, bounds: list) -> list:
    """
    Divide bounds into n uniformly distributed points

    Args:
        n (int): # divisions
        bounds (list[float]): array bounds [start, end]
        
    Returns:
        list: n-size array of floats
    """
    start = bounds[0]
    end = bounds[1]
    
    return stats.uniform.rvs(loc=start, scale=end-start, size=n)

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

def pol2rect(pol: list) -> list:
    """
    Convert from polar/spherical to rectangular coordinates

    Args:
        pol (list[int]): polar/spherical coordinates
        
    Returns:
        list: rectangular coordinates
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

def rect2pol(rect: list) -> list:
    """
    Convert from rectangular to polar/spherical coordinates

    Args:
        rect (list): rectangular coordinates

    Returns:
        list: spherical coordinates
    """
    r = np.linalg.norm(rect)
    if rect[0] != 0:
        theta_acute = np.arctan(rect[1]/rect[0])
    else:
        theta_acute = np.pi
    
    if np.sign(rect[0]) > 0:
        if np.sign(rect[1]) >= 0:
            theta = theta_acute
        elif np.sign(rect[1]) < 0:
            theta = twopi + theta_acute
    elif np.sign(rect[0]) < 0:
        theta = np.pi + theta_acute
    else:
        if np.sign(rect[1]) > 0:
            theta = halfpi
        else:
            theta = 3*halfpi
    
    if len(rect) == 3:
        phi = np.arccos(rect[2]/r)
        return np.array([r, theta, phi])
    
    return np.array([r, theta])

def angle2bearing(angle: float) -> float:
    """
    Convert angle (unit circle) to bearing

    Args:
        angle (float): angle mod 2*pi in radians

    Returns:
        float: bearing in radians, N = y axis
    """
    angle = twopi - angle
    bearing = (angle + halfpi) % twopi
    
    return bearing  

def bearing2angle(bearing: float) -> float:
    """
    Convert bearing to angle (unit circle)

    Args:
        bearing (float): bearing in radians, N = y axis

    Returns:
        float: angle (unit circle) mod 2*pi in radians
    """
    bearing = twopi - bearing
    angle = (bearing + halfpi) % twopi
    
    return angle

def line_plane_acute(line: list, normal: list) -> float:
    """
    Calculate acute angle between a line and a plane in 3D

    Args:
        line (list): [x,y,z] of line
        normal (list): [x,y,z] of plane's normal

    Returns:
        float: acute angle in radians
    """
    mag_line = np.linalg.norm(line)
    mag_normal = np.linalg.norm(normal)
    comp = np.arccos(np.dot(line, normal)/(mag_line*mag_normal))
    if comp > halfpi: comp = np.pi - comp
    
    return halfpi - comp
    
def plane_plane_acute(normal_1: list, normal_2: list) -> float:
    """
    Calculate acute angle between two planes in 3D

    Args:
        normal_1 (list): [x,y,z] of 1st plane's normal
        normal_2 (list): [x,y,z] of 2nd plane's normal

    Returns:
        float: acute angle in radians
    """
    mag_1 = np.linalg.norm(normal_1)
    mag_2 = np.linalg.norm(normal_2)
    angle = np.arccos(np.dot(normal_1, normal_2)/(mag_1*mag_2))
    if angle > halfpi: angle = np.pi - angle
    
    return angle

def sphere_to_sd(point: list) -> list:
    """
    Find the strike-dip pair corresponding to a point on the surface of
    an upper hemisphere of unit radius

    Args:
        point (list): [x,y,z] coordinates of point (normal to the fault plane)

    Returns:
        list: [s,d] pair corresponding to the sphere point
    """    
    spherical = rect2pol(point)
    plane_angle = (spherical[1] + halfpi) % twopi
    strike = angle2bearing(plane_angle)
    dip = spherical[2]
    
    return np.array([strike, dip])

def sd_to_sphere(sd: list) -> list:
    """
    Find a point on the surface of an upper hemisphere corresponding to
    input strike-dip pair

    Args:
        sd (list): [strike, dip] pair

    Returns:
        list: [x,y,z] coordinates
    """
    r = 1
    normal_bearing = (sd[0] + halfpi) % twopi
    theta = bearing2angle(normal_bearing)
    phi = sd[1]
    
    return pol2rect([r, theta, phi])

def starting_direc(point: list, direc: list) -> list:
    """
    Find what vector is perpendicular to point and lies in the
    vertical plane

    Args:
        point (list): [x,y,z] on a unit sphere
        direc (list): direction to match, unit length

    Returns:
        list: [x,y,z] of the normal vector
    """
    proj = (np.dot(direc,point)/np.dot(point, point))
    rem = np.array(direc) - np.array([proj*p for p in point])
    
    return rem/np.linalg.norm(rem)

def rotate(vec: list, axis: list, theta: float) -> list:
    """
    Rotate vec about axis

    Args:
        vec (list): vector to be rotated
        axis (list): axis of rotation, has unit length
        theta (float): angle of rotation

    Returns:
        list: [x,y,z] of resultant vector
    """
    ux, uy, uz = axis
    
    row1 = [np.cos(theta) + (ux**2)*(1-np.cos(theta)),
            uy*ux*(1-np.cos(theta)) + uz*np.sin(theta),
            uz*ux*(1-np.cos(theta)) - uy*np.sin(theta)]
    
    row2 = [ux*uy*(1-np.cos(theta)) - uz*np.sin(theta),
            np.cos(theta) + (uy**2)*(1-np.cos(theta)),
            uz*uy*(1-np.cos(theta)) + ux*np.sin(theta)]
    
    row3 = [ux*uz*(1-np.cos(theta)) + uy*np.sin(theta),
            uy*uz*(1-np.cos(theta)) - ux*np.sin(theta),
            np.cos(theta) + (uz**2)*(1-np.cos(theta))]
    
    R = [row1, row2, row3]
    
    return np.matmul(vec, R)

def azdp(v, units):
    """
    From Suzan, different from Omkar's code
    IN: vector in r (up), theta (south), and phi (east) coordinates
    OUT: azimuth and dip (plunge) of vector 
    """
    vr, vt, vp = v
    dparg = np.sqrt(vt*vt + vp*vp)
    if dparg < eps:
        vaz = 0.
        vdp = np.pi
    elif abs(vr) < eps:
        vaz = np.pi - np.arctan2(vp,vt) 
        vdp = 0.
    else:
        vaz = np.pi - np.arctan2(vp,vt) 
        vdp = np.arctan2(-vr,dparg)

    if units == 'deg':
        vdp = np.degrees(vdp)
        vaz = np.degrees(vaz)
    return vaz,vdp  # in specified units (degrees or radians)

def coord_switch(point: list) -> list:
    """
    Change coordinate system to fit tp2sdr function

    Args:
        point (list): [x,y,z] coordinates

    Returns:
        list: new [x,y,z] coordinates
    """
    return np.array([point[2], -point[1], point[0]])

def tp2sdr(t,p):
    """
    From Suzan, tweaked to keep strike in [0, 2*pi)
    Use rectangular coordinates, x is up, y is south and z is east
    IN: vectors representing T and P axes (in r-theta-phi (normal mode) coordinates)
        reminder: r (up), theta (south), and phi (east)
    OUT: strike, dip, and rake for both fault planes (double couple assumed)
    """
    pole1 = (t+p)/np.sqrt(2.)
    pole2 = (t-p)/np.sqrt(2.)
    az1,dp1 = azdp(pole1,'rad')
    az2,dp2 = azdp(pole2,'rad')

    st1 = az1 - halfpi
    dip1 = dp1 + halfpi
    x = np.array([0.,-1*np.cos(st1),np.sin(st1)])
    cosrake = np.dot(x, pole2)
    if np.abs(cosrake) > 1:
        cosrake = np.sign(cosrake)
    if pole2[0] > 0:
        rake1 = np.arccos(cosrake)
    else:
        rake1 = -np.arccos(cosrake)
    if pole1[0] < 0.:
        st1 = st1 + np.pi
        dip1 = np.pi - dip1
        rake1 = -rake1
    if (np.cos(dp1) < eps):
        st1 = 0.
        dip1 = 0.
        if pole2[0] > 0.:
           rake1 = -az2
        else:
           rake1 = -az2 + np.pi

    st2 = az2 - halfpi
    dip2 = dp2 + halfpi
    x = np.array([0.,-1*np.cos(st2),np.sin(st2)])
    cosrake = np.dot(x, pole1)
    if np.abs(cosrake) > 1:
        cosrake = np.sign(cosrake)
    if pole1[0] > 0.:
        rake2 = np.arccos(cosrake)
    else:
        rake2 = -np.arccos(cosrake)
    if pole2[0] < 0.:
        st2 = st2 + np.pi
        dip2 = np.pi - dip2
        rake2 = -rake2
    if (np.cos(dp2) < eps):
        st2 = 0.
        dip2 = 0.
        if pole2[0] > 0.:
           rake2 = -az1
        else:
           rake2 = -az1 + np.pi

    return (st1%twopi, dip1, rake1), (st2%twopi, dip2, rake2)  # in radians
    
def Rpattern(fault, azimuth, takeoff_angles): # Update Omkar on incidence -> takeoff
    """
    Calculate predicted amplitudes of P, SV, and SH waves.
    IN: fault = [strike, dip, rake]
             = faulting mechanism, described by a list of strike, dip, and rake
             (note, strike is measured clockwise from N, dip is measured positive downwards
             (between 0 and 90) w.r.t. a horizontal that is 90 degrees clockwise from strike,
             and rake is measured positive upwards (counterclockwise)
        azimuth: azimuth with which ray path leaves source (clockwise from N)
        takeoff_angles = [i, j]
              i = angle between P ray path & vertical in the source model layer
              j = angle between S ray path & vertical in the source model layer
    OUT: Amplitudes for P, SV, and SH waves
    P as measured on L (~Z) component, SV measured on Q (~R) component, and SH measured on T component.
    All input is in degrees. 
    (c) 2020 Suzan van der Lee
    """

    strike, dip, rake = fault
    a = azimuth; rela = strike - azimuth
    sinlam = np.sin(np.radians(rake))
    coslam = np.cos(np.radians(rake))
    sind = np.sin(np.radians(dip))
    cosd = np.cos(np.radians(dip))
    cos2d = np.cos(np.radians(2*dip))
    sinrela = np.sin(np.radians(rela))
    cosrela = np.cos(np.radians(rela))
    sin2rela = np.sin(np.radians(2*rela))
    cos2rela = np.cos(np.radians(2*rela))

    sR = sinlam*sind*cosd
    qR = sinlam*cos2d*sinrela + coslam*cosd*cosrela
    pR = coslam*sind*sin2rela - sinlam*sind*cosd*cos2rela
    pL = sinlam*sind*cosd*sin2rela + coslam*sind*cos2rela
    qL = -coslam*cosd*sinrela + sinlam*cos2d*cosrela

    iP = np.radians(takeoff_angles[0])
    jS = np.radians(takeoff_angles[1])

    AP = sR*(3*np.cos(iP)**2 - 1) - qR*np.sin(2*iP) - pR*np.sin(iP)**2
    ASV = 1.5*sR*np.sin(2*jS) + qR*np.cos(2*jS) + 0.5*pR*np.sin(2*jS)
    ASH = qL*np.cos(jS) + pL*np.sin(jS)

    return AP,ASV,ASH

def mag_perc(u: list, v: list) -> float:
    """
    Calculate the magnitude of the component of vector u
    that is perpendicular to vector v
    
    Args:
        u (list): vector u
        v (list): vector v

    Returns:
        float: magnitude of the perpendicular component of u
    """
    return np.linalg.norm(np.cross(u,v))/np.linalg.norm(v)

def get_sphere_epsilon(Ao: list, Uo: list) -> float:
    """
    Calculate epsilon, the toletance angle given the observed amplitudes
    and their uncertainties
    Args:
        Ao (list): observed amplitudes
        Uo (list): uncertainty of observed amplitudes

    Returns:
        float: epsilon in radians
    """
    sig1 = np.array([Uo[0],0,0])
    sig2 = np.array([0,Uo[1],0])
    sig3 = np.array([0,0,Uo[2]])
    sig = [sig1,sig2,sig3]
    e = np.sqrt((1/3)*np.sum([mag_perc(sig[i],Ao)**2 for i in range(len(Ao))]))
    epsilon = np.arctan(e/np.linalg.norm(Ao))
    
    return epsilon

def get_ellipse_epsilon(Ao: list, Uo: list, As: list) -> float:
    """
    Calculate epsilon, the toletance angle given the observed amplitudes
    Accounts for the ellipsoid's asymmetry
    Inputs are numpy arrays

    Args:
        Ao (list): observed amplitudes
        Uo (list): uncertainty of observed amplitudes
        As (list): simulated amplitudes

    Returns:
        float: epsilon in radians
    """
    
    n1 = np.cross(Ao,As)
    n2 = Ao/(Uo**2)
    m = np.cross(n1,n2)
    v = m/Uo
    k = 1 - 1/np.dot(n2,Ao)
    b = np.dot(n2,m)/np.dot(n2,Ao)
    
    t1 = (b - np.sqrt(b**2 + k*np.dot(v,v)))/np.dot(v,v)
    t2 = (b + np.sqrt(b**2 + k*np.dot(v,v)))/np.dot(v,v)
    
    r1 = k*Ao + t1*np.cross(n1,n2)
    r2 = k*Ao + t2*np.cross(n1,n2)
    
    epsilon = .5*np.arccos(np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))
    
    return epsilon

def get_gaussian_weight(angle: float, epsilon: float) -> float:
    """
    Calculate the gaussian weight given the angle and epsilon
    Angles are in radians
    
    Args:
        angle (float): angle between the observed and predicted amplitudes
        epsilon (float): tolerance angle, one standard deviation of the gaussian
        
    Returns:
        float: gaussian weight
    """
    return np.exp(-angle**2/(2*epsilon**2))

def apply_inverse_methods(N: int, hdepth: float, epdist: float, azimuth: float,
                          Ao: list, Uo: list, model: TauPyModel) -> pd.DataFrame:
    """
    Generate a dataframe with appropriate weights attached to each simulated
    focal mechanism

    Args:
        N (int): Number of rake simulations per s-d pair
        hdepth (float): assumed hypocentral depth in km
        epdist (float): epicentral distance in degrees
        azimuth (float): quake azimuth in degrees
        Ao (list): observed amplitudes
        Uo (list): uncertainty of observed amplitudes
        model (TauPyModel): velocity model
        
    Returns:
        pd.DataFrame: dataframe with weights attached to each simulated focal mechanism
        Columns: ["Theta", "Phi", "Alpha", "Strike1", "Dip1", "Rake1", "Strike2", "Dip2", "Rake2", "OldWeight", "Weight"]
        All angles are in degrees
        Theta, phi are spherical coordinates of the normal vector t
        Alpha is the rotation angle of the normal vector p
    """
    b3_over_a3 = (3.4600/5.8000)**3 # from Suzan, not part of the velocity model
    arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                        distance_in_degree=epdist, phase_list=['P', 'S'])
    takeoff_angles = [a.takeoff_angle for a in arrivals]
    
    data = collections.defaultdict(list) # initialize dataframe
    
    old_epsilon = get_sphere_epsilon(Ao, Uo)
    t_samples = hemisphere_samples(N**2)
    for t in t_samples:
        p_rotations = uniform_samples(N, [0, np.pi])
        p_start = starting_direc(t, j_hat)
        for alpha in p_rotations:
            r, theta, phi = rect2pol(t)
            p = rotate(p_start, t, alpha)
            sdr1, sdr2 = tp2sdr(coord_switch(t), coord_switch(p))
            sdr1 = np.rad2deg(np.array(sdr1))
            sdr2 = np.rad2deg(np.array(sdr2))
            As = np.array(Rpattern(sdr1, azimuth, takeoff_angles))
            As[0] *= b3_over_a3
            angle = np.arccos(np.dot(As, Ao)/(np.linalg.norm(As)*np.linalg.norm(Ao)))
            
            old_weight = get_gaussian_weight(angle, old_epsilon) # old method
            epsilon = get_ellipse_epsilon(Ao, Uo, As)
            weight = get_gaussian_weight(angle, epsilon) # new method
            data["Theta"].append(np.rad2deg(theta))
            data["Phi"].append(np.rad2deg(phi))
            data["Alpha"].append(np.rad2deg(alpha))
            data["Strike1"].append(sdr1[0])
            data["Dip1"].append(sdr1[1])
            data["Rake1"].append(sdr1[2])
            data["Strike2"].append(sdr2[0])
            data["Dip2"].append(sdr2[1])
            data["Rake2"].append(sdr2[2])
            data["OldWeight"].append(old_weight)
            data["Weight"].append(weight)
    
    return pd.DataFrame(data)

def plot_sdr_histograms(df: pd.DataFrame, bins: int):
    """
    Plot histograms of the sdr pairs

    Args:
        df (pd.DataFrame): dataframe with sdr pairs
        bins (int): number of bins
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    plt.suptitle("SDR Histograms")
    axs[0, 0].hist(df["Strike1"], bins=bins)
    axs[0, 0].set_title("Strike 1")
    axs[1, 0].hist(df["Dip1"], bins=bins)
    axs[1, 0].set_title("Dip 1")
    axs[1,0].set_ylabel("Frequency")
    axs[2, 0].hist(df["Rake1"], bins=bins)
    axs[2, 0].set_title("Rake 1")
    axs[2,0].set_xlabel("Degrees")
    axs[0, 1].hist(df["Strike2"], bins=bins)
    axs[0, 1].set_title("Strike 2")
    axs[1, 1].hist(df["Dip2"], bins=bins)
    axs[1, 1].set_title("Dip 2")
    axs[2, 1].hist(df["Rake2"], bins=bins)
    axs[2, 1].set_title("Rake 2")
    axs[2,1].set_xlabel("Degrees")
    plt.show()
    
def weighted_3D_scatter(df: pd.DataFrame, weight: str):
    """
    Plot a 3D scatter plot of the sdr pairs, weighted by the specified column

    Args:
        df (pd.DataFrame): dataframe with sdr pairs
        title (str): plot title
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    if weight == "OldWeight":
        ax.set_title("Weighted 3D Scatter Plot (Old Method)")
    else:
        ax.set_title("Weighted 3D Scatter Plot (New Method)")
    ax.set_xlabel("Rake")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Dip")
    scatter = ax.scatter(df["Rake1"]._append(df["Rake2"]),
                         df["Strike1"]._append(df["Strike2"]),
                         df["Dip1"]._append(df["Dip2"]),
                         c=df[weight]._append(df[weight]), cmap="YlGnBu")
    plt.colorbar(scatter)
    plt.show()

    """
    Check out distribution of entire dataset...
    Where does the S-wave come from?
    Add true solution to weighted 3D scatter plot
    Plot the true solution as a red star
    New function for 2D scatters/histograms
    Learn how to plot beachballs
    """
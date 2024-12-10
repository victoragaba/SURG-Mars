import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import collections
import math

from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import beach

eps = 1e-10; halfpi = np.pi/2; twopi = 2*np.pi
i_hat = np.array([1,0,0]); j_hat = np.array([0,1,0]); k_hat = np.array([0,0,1])
np.random.seed(1029) # I hope this carries over to other files
# It only does for randomness associated with fr

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

def random_hemisphere_samples(n: int) -> list:
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
        while np.linalg.norm(p) < eps or all(p[:2]) == 0:
            p = stats.norm.rvs(size=3)
        p /= np.linalg.norm(p)
        p[2] = np.abs(p[2]) # only upper hemisphere
        samples.append(p)
    
    return samples

def rigid_hemisphere_samples(step_size: float) -> list:
    """
    Systematically select n samples evenly spaced on the surface of an upper hemisphere

    Args:
        step_size (float): angular distance between samples in degrees

    Returns:
        list: array of [x,y,z] coordinates
    """
    samples = [k_hat]
    d_phi = np.deg2rad(step_size)
    phis = np.arange(0, halfpi, d_phi)
    
    for phi in phis[1:]:
        c_phi = twopi*np.sin(phi)
        thetas = np.linspace(0, twopi, int(c_phi/d_phi)+1)[:-1]
        samples.extend([pol2rect(np.array([1, theta, phi])) for theta in thetas])
        
    return samples

def suzan_hemisphere_samples(step_size: float) -> list:
    """
    Systematically select n samples evenly spaced on the surface of an upper hemisphere
    Suzan's method

    Args:
        step_size (float): angular distance between samples in degrees

    Returns:
        list: array of [x,y,z] coordinates
    """
    dd = np.deg2rad(step_size)
    angles = np.arange(0,np.pi,dd)
    Ts = [np.array([-1,0,0])] # southpole
    # sphcs = [np.array([0,0])]
    
    for a in angles[1:]:    # a is "take-off angle" of T axis
                        #(measured w.r.t. downpointing vertical) ("latitude")
        ddo = dd/np.sin(a)  # need fewer sampling points on small-circle when close to the poles
        angleso = np.arange(0,np.pi,ddo)
        for o in angleso:   # o is azimuth of T axis ("longitude")
            T = np.array([-np.cos(a), -np.sin(a)*np.cos(o), np.sin(a)*np.sin(o)])
            Ts.append(T)
            # sphcs.append(np.degrees(np.array([o,a])))
    
    return Ts

def perp(vin):
    """
    IN: a vector (e.g. a T axis) 
    Rotate a vector over 90 degrees around any axis
    OUT: rotated vector of the same length as the input vector
    """
    vinl = np.linalg.norm(vin)
    l = np.argmin(np.abs(vin))
    m = np.argmax(np.abs(vin))
    k = 0; j = 1; i = 2
    if k != l:
        if j != l:
            i = k
            k = l
        else:
            j = k
            k = l     
    if j != m:
        i = j
        j = m
    v = np.zeros(3)
    v[k] = 0
    v[i] = 1
    v[j] = -vin[i]/vin[j]
    vl = np.linalg.norm(v)
    
    return v*vinl/vl

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
    # r = np.linalg.norm(rect)
    # if rect[0] != 0:
    #     theta_acute = np.arctan(rect[1]/rect[0])
    # else:
    #     theta_acute = np.pi
    
    # if np.sign(rect[0]) > 0:
    #     if np.sign(rect[1]) >= 0:
    #         theta = theta_acute
    #     elif np.sign(rect[1]) < 0:
    #         theta = twopi + theta_acute
    # elif np.sign(rect[0]) < 0:
    #     theta = np.pi + theta_acute
    # else:
    #     if np.sign(rect[1]) > 0:
    #         theta = halfpi
    #     else:
    #         theta = 3*halfpi
    
    # if len(rect) == 3:
    #     phi = np.arccos(rect[2]/r)
    #     return np.array([r, theta, phi])
    
    # return np.array([r, theta])
    
    if len(rect) == 2:
        x, y = rect
        r = np.linalg.norm(rect)
        theta = np.arctan2(y, x) % twopi
        return np.array([r, theta])
    
    x, y, z = rect
    rho = np.linalg.norm(rect)
    theta = np.arctan2(y, x) % twopi
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    return np.array([rho, theta, phi])

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
    Find what vector is perpendicular to point vector and lies in the
    plane that contains point and direc

    Args:
        point (list): [x,y,z] on a unit sphere
        direc (list): direction to match, unit length

    Returns:
        list: [x,y,z] of the normal vector
    """
    # # Let's get rotational
    # if np.array_equal(point, direc):
    #     print("That one exception")
    #     return j_hat # that one exception
    # out = rodrigues_rotate(point, direc, halfpi)
    # return out/np.linalg.norm(out)
    
    # Let's get linear algebraic
    proj = (np.dot(direc, point)/np.dot(point, point)) # parallel to point
    rem = np.array(direc) - proj*np.array(point) # orthogonal to point
    return rem/np.linalg.norm(rem) if np.linalg.norm(rem) > eps else j_hat

def matrix_rotate(vec: list, axis: list, theta: float) -> list:
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

def rodrigues_rotate(vec: list, axis: list, theta: float) -> list:
    """
    Rotate vec around axis by theta radians
    Assumes axis is normalized
    """
    rotated_vec = (
        vec*np.cos(theta) + 
        (np.cross(axis,vec))*np.sin(theta) + 
        axis*(np.dot(axis,vec))*(1-np.cos(theta))
    )
    
    return rotated_vec

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
    First coordinate points up, second points south, third points east

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

# my personal favorite
def my_tp2sdr(t: list, p: list, deg: bool = False) -> tuple:
    """
    Converts from T and P axes to strike, dip, and rake of both fault planes
    Inputs are numpy arrays, assumes unit vectors
    Uses right-hand rule for coordinate system

    Args:
        t (list): T axis
        p (list): P axis
        deg (bool, optional): output in degrees if True

    Returns:
        tuple: (sdr1)
    """
    # First get the normals
    n1 = t + p
    n2 = t - p
    
    # Restrict to upper hemisphere
    if n1[2] < 0: n1 *= -1
    if n2[2] < 0: n2 *= -1
    
    # Get spherical coordinates, dip is restricted phi
    _, theta1, dip1 = rect2pol(n1)
    _, theta2, dip2 = rect2pol(n2)
    
    # Strike is theta measured in reverse
    strike1 = (twopi - theta1) % twopi
    strike2 = (twopi - theta2) % twopi
    
    # Base directions for rakes, null for exceptions
    base1 = pol2rect(np.array([1, bearing2angle(strike1), halfpi]))
    base2 = pol2rect(np.array([1, bearing2angle(strike2), halfpi]))
    null = np.cross(t,p)
    if null[2] < 0: null *= -1
    done1, done2 = False, False
    
    # Account for edge cases on perfectly vertical dips
    if n1[2] < eps and n2[2] < eps: # double vertical dip
        check1 = rodrigues_rotate(base1, null, np.pi/4)
        if abs(np.dot(check1, t)) < eps: rake1 = 0.
        else: rake1 = np.pi
        check2 = rodrigues_rotate(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done1, done2 = True, True
    elif n1[2] < eps: # single vertical dip
        check2 = rodrigues_rotate(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done2 = True
    elif n2[2] < eps: # single vertical dip
        check1 = rodrigues_rotate(base1, null, np.pi/4)
        if abs(np.dot(check1, t)) < eps: rake1 = 0.
        else: rake1 = np.pi
        done1 = True
    
    # Check if the restricted normals house a t or p axis
    if abs(np.dot(n1 + n2, t)) < eps: # houses a p axis, negative slip
        slip1, slip2 = -n2, -n1
    else: # houses a t axis, positive slip
        slip1, slip2 = n2, n1

    # Get rakes from  base and slip vectors
    if not done1:
        rake1 = np.arccos(np.dot(slip1, base1)/np.linalg.norm(slip1))
        rake1 *= np.sign(slip1[2])
    if not done2:
        rake2 = np.arccos(np.dot(slip2, base2)/np.linalg.norm(slip2))
        rake2 *= np.sign(slip2[2])
    
    if deg:
        strike1, strike2 = np.rad2deg(strike1), np.rad2deg(strike2)
        dip1, dip2 = np.rad2deg(dip1), np.rad2deg(dip2)
        rake1, rake2 = np.rad2deg(rake1), np.rad2deg(rake2)
    
    return (np.array([strike1, dip1, rake1]), np.array([strike2, dip2, rake2]))

# Incorporate b3_over_a3 in this function (alpha, beta)  
def Rpattern(fault, azimuth, takeoff_angles, alpha_beta: list = []): # Update Omkar on incidence -> takeoff
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
    rela = strike - azimuth
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
    
    # Incorporating b3_over_a3
    if len(alpha_beta) != 0:
        AP /= alpha_beta[0]**3
        ASV /= alpha_beta[1]**3
        ASH /= alpha_beta[1]**3

    return np.array([AP,ASV,ASH])

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
    if np.linalg.norm(n1) < eps: # can't use As
        if np.dot(Ao, As) > 0: return np.inf
        else: return eps
    n2 = Ao/(Uo**2)
    
    m = np.cross(n1,n2)
    v = m/Uo
    k = 1 - 1/np.dot(n2,Ao)
    b = np.dot(n2,m)/np.dot(n2,Ao)
    
    vdot = np.dot(v,v)
    t1 = (b - np.sqrt(b**2 + k*vdot))/vdot
    t2 = (b + np.sqrt(b**2 + k*vdot))/vdot
    
    r1 = k*Ao + t1*m
    r2 = k*Ao + t2*m
    
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

# Incorporate b3_over_a3 into Rpattern
def apply_inverse_methods(model: TauPyModel, t_vec: str, p_vec: str, sdr1_vec: str, sdr2_vec: str, hdepth: float, epdist: float, azimuth: float,
                          Ao: list, Uo: list, no_rejected: bool = False) -> pd.DataFrame:
    """
    Generate a dataframe with appropriate weights attached to each simulated
    focal mechanism

    Args:
        t_vec, p_vec, sdr1, sdr2 (str): vectors of t, p, and sdr components
        hdepth (float): assumed hypocentral depth in km
        epdist (float): epicentral distance in degrees
        azimuth (float): quake azimuth in degrees
        Ao (list): observed amplitudes
        Uo (list): uncertainty of observed amplitudes
        model (TauPyModel): velocity model
        no_rejected (bool, optional): whether to include rejected mechanisms. Defaults to False.
        
    Returns:
        pd.DataFrame: dataframe with weights attached to each simulated focal mechanism
        Columns: ["Theta", "Phi", "Alpha", "Strike1", "Dip1", "Rake1", "Strike2", "Dip2", "Rake2", "OldWeight", "Weight"]
        All angles are in degrees
        Theta, phi are spherical coordinates of the normal vector t
        Alpha is the rotation angle of the normal vector p
    """
    # b3_over_a3 = (3.4600/5.8000)**3
    alpha_beta = [5.8000, 3.4600] # incorporated into Rpattern
    arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                        distance_in_degree=epdist, phase_list=['P', 'S'])
    takeoff_angles = [a.takeoff_angle for a in arrivals]
    
    data = collections.defaultdict(list) # initialize dataframe
    
    old_epsilon = get_sphere_epsilon(Ao, Uo)
    t_samples = np.load(t_vec)
    p_samples = np.load(p_vec)
    sdr1_samples = np.load(sdr1_vec)
    sdr2_samples = np.load(sdr2_vec)
    
    for i in range(len(t_samples)):
        _, theta, phi = rect2pol(t_samples[i])
        sdr1, sdr2 = sdr1_samples[i], sdr2_samples[i]
        As = Rpattern(sdr1, azimuth, takeoff_angles, alpha_beta)
        # As[0] *= b3_over_a3
        angle = np.arccos(np.dot(As, Ao)/(np.linalg.norm(As)*np.linalg.norm(Ao)))
        
        old_weight = get_gaussian_weight(angle, old_epsilon) # old method
        epsilon = get_ellipse_epsilon(Ao, Uo, As)
        weight = get_gaussian_weight(angle, epsilon) # new method
        if no_rejected:
            if old_weight < np.exp(-2) and weight < np.exp(-2): continue
        data["Theta"].append(np.rad2deg(theta))
        data["Phi"].append(np.rad2deg(phi))
        data["tx"].append(t_samples[i][0])
        data["ty"].append(t_samples[i][1])
        data["tz"].append(t_samples[i][2])
        data["px"].append(p_samples[i][0])
        data["py"].append(p_samples[i][1])
        data["pz"].append(p_samples[i][2])
        data["Strike1"].append(sdr1[0])
        data["Dip1"].append(sdr1[1])
        data["Rake1"].append(sdr1[2])
        data["Strike2"].append(sdr2[0])
        data["Dip2"].append(sdr2[1])
        data["Rake2"].append(sdr2[2])
        data["AP"].append(As[0])
        data["ASV"].append(As[1])
        data["ASH"].append(As[2])
        data["OldWeight"].append(old_weight)
        data["Weight"].append(weight)
    
    return pd.DataFrame(data)

# Core function
def sdr_inverse_method(sdr_vec: str, azimuth: float, takeoffs: list, Ao: list, Uo: list, alpha_beta: list = [5.8000, 3.4600], no_rejected: bool = False) -> pd.DataFrame:
    sdrs = np.load(sdr_vec)
    data = collections.defaultdict(list) # initialize dataframe
    
    for sdr in sdrs:
        As = Rpattern(sdr, azimuth, takeoffs, alpha_beta)
        angle = np.arccos(np.dot(As, Ao)/(np.linalg.norm(As)*np.linalg.norm(Ao)))
        old_epsilon = get_sphere_epsilon(Ao, Uo)
        old_weight = get_gaussian_weight(angle, old_epsilon) # old method
        epsilon = get_ellipse_epsilon(Ao, Uo, As)
        weight = get_gaussian_weight(angle, epsilon) # new method
        if no_rejected:
            if old_weight < np.exp(-2) and weight < np.exp(-2): continue
        data['Strike'].append(sdr[0])
        data['Dip'].append(sdr[1])
        data['Rake'].append(sdr[2])
        data['AP'].append(As[0])
        data['ASV'].append(As[1])
        data['ASH'].append(As[2])
        data['OldWeight'].append(old_weight)
        data['Weight'].append(weight)
        
    return pd.DataFrame(data)

def sdr_histograms(df: pd.DataFrame, bins: int = 50):
    """
    Plot histograms of the sdr pairs

    Args:
        df (pd.DataFrame): dataframe with sdr pairs
        bins (int): number of bins. Defaults to 50.
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
    
def weighted_3D_scatter(df: pd.DataFrame, weight: str, true_sol: list = [], type: str = "sdr", save: str = ""):
    """
    Plot a 3D scatter plot of the sdr triples or amplitudes, weighted by the specified column

    Args:
        df (pd.DataFrame): dataframe with sdr pairs
        weight (str): method of weighting
        true_sol (list, optional): true solution, NumPy array of sdr*2. Defaults to [].
        type (str, optional): "sdr" or "amp" (amplitude). Defaults to "sdr".
    """
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    if weight == "OldWeight":
        ax.set_title(f"Weighted {type} Scatter Plot (Old Method)", fontsize=20)
    elif weight == "Weight":
        ax.set_title(f"Weighted {type} Scatter Plot (New Method)", fontsize=20)
    else:
        ax.set_title(f"{type} Scatter Plot", fontsize=20)
    
    if type == "amp":
        labels = ["AP", "ASV", "ASH"]
        scatter = ax.scatter(df["AP"], df["ASV"], df["ASH"], c=df[weight], cmap="YlGnBu")
    elif type == "sdr":
        labels = ["Rake", "Strike", "Dip"]
        scatter = ax.scatter(df["Rake1"]._append(df["Rake2"]),
                        df["Strike1"]._append(df["Strike2"]),
                        df["Dip1"]._append(df["Dip2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
    else:
        print("Invalid type")
        return
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    
    plt.colorbar(scatter)
    
    if true_sol != [] and type == "sdr":
        true_sol = np.transpose(true_sol)
        ax.scatter(true_sol[2], true_sol[0], true_sol[1], c='red', marker='o', s=500)
    if save != "":
        plt.savefig(save)
    plt.show()
   
def weighted_pairwise_scatter(df: pd.DataFrame, weight: str, bins: int = 50, true_sol: list = [], type: str = "sdr"):
    """
    Plot pairwise scatter plots of the sdr, tp or amplitude pairs, weighted by the specified column
    Alternatively histograms 

    Args:
        df (pd.DataFrame): dataframe with sdr pairs
        weight (str): method of weighting
        bins (int): array of bins. Defaults to 50.
        true_sol (list, optional): true solution, NumPy array of two sdr triples. Defaults to [].
        type (str, optional): "sdr" or "tp" (theta, phi) or "amp" (amplitudes). Defaults to "sdr".
    """
    if type == "tp":
        # Plot histograms for theta, phi, alpha
        fig, axs = plt.subplots(1, 3, figsize=(20,5))
        if weight == "OldWeight":
            plt.suptitle("Theta, Phi Visuals (Old Method)", fontsize=20)
        else:
            plt.suptitle("Theta, Phi Visuals (New Method)", fontsize=20)
        
        axs[0].hist(df["Theta"], bins=np.linspace(0,360,bins))
        axs[0].set_title("Theta Histogram")
        axs[0].set_xlabel("Degrees")
        axs[0].set_ylabel("Frequency")
        
        axs[1].hist(df["Phi"], bins=np.linspace(0,90,bins))
        axs[1].set_title("Phi Histogram")
        axs[1].set_xlabel("Degrees")
        axs[1].set_ylabel("Frequency")
        
        if weight == "OldWeight":
            axs[2].set_title("Weighted Theta-Phi Scatter Plot (Old Method)")
        else:
            axs[2].set_title("Weighted Theta-Phi Scatter Plot (New Method)")
        axs[2].set_xlabel("Theta")
        axs[2].set_ylabel("Phi")
        scatter = axs[2].scatter(df["Theta"], df["Phi"], c=df[weight], cmap="YlGnBu", label='weight')
        axs[2].legend()
        plt.colorbar(scatter)
        if true_sol != []:
            true_sol = np.transpose(true_sol)
            axs[2].scatter(true_sol[0], true_sol[1], c='red', marker='o', s=500)
        
        plt.show()
        
    elif type == "sdr":
    
        if true_sol != []: true_sol = np.transpose(true_sol)
        # Use similar structure as sdr_histograms
        # Don't stack
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        if weight == "OldWeight":
            plt.suptitle("Weighted Pairwise Scatter Plots (Old Method)", fontsize=20)
        else:
            plt.suptitle("Weighted Pairwise Scatter Plots (New Method)", fontsize=20)
        
        axs[0,0].hist(df["Strike1"]._append(df["Strike2"]), bins=np.linspace(0,360,bins))
        if true_sol != []:
            axs[0,0].axvline(x=true_sol[0][0], color='green', label='Strike1')
            axs[0,0].axvline(x=true_sol[0][1], color='red', label='Strike2')
            axs[0,0].legend()
        axs[0,0].set_title("Strike")
        axs[0,0].set_xlabel("Degrees")
        axs[0,0].set_ylabel("Frequency")
        
        axs[0,1].scatter(df["Dip1"]._append(df["Dip2"]),
                        df["Strike1"]._append(df["Strike2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[0,1].set_title("Dip vs. Strike")
        axs[0,1].set_xlabel("Dip (Degrees)")
        axs[0,1].set_ylabel("Strike (Degrees)")
        if true_sol != []:
            axs[0,1].scatter(true_sol[1], true_sol[0], c='red', marker='o', s=100)
        
        axs[0,2].scatter(df["Rake1"]._append(df["Rake2"]),
                        df["Strike1"]._append(df["Strike2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[0,2].set_title("Rake vs. Strike")
        axs[0,2].set_xlabel("Rake (Degrees)")
        axs[0,2].set_ylabel("Strike (Degrees)")
        if true_sol != []:
            axs[0,2].scatter(true_sol[2], true_sol[0], c='red', marker='o', s=100)
        
        axs[1,0].scatter(df["Strike1"]._append(df["Strike2"]),
                        df["Dip1"]._append(df["Dip2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[1,0].set_title("Strike vs. Dip")
        axs[1,0].set_xlabel("Strike (Degrees)")
        axs[1,0].set_ylabel("Dip (Degrees)")
        if true_sol != []:
            axs[1,0].scatter(true_sol[0], true_sol[1], c='red', marker='o', s=100)
        
        axs[1,1].hist(df["Dip1"]._append(df["Dip2"]), bins=np.linspace(0,90,bins))
        if true_sol != []:
            axs[1,1].axvline(x=true_sol[1][0], color='green', label='Dip1')
            axs[1,1].axvline(x=true_sol[1][1], color='red', label='Dip2')
            axs[1,1].legend()
        axs[1,1].set_title("Dip")
        axs[1,1].set_xlabel("Degrees")
        axs[1,1].set_ylabel("Frequency")
        
        axs[1,2].scatter(df["Rake1"]._append(df["Rake2"]),
                        df["Dip1"]._append(df["Dip2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[1,2].set_title("Rake vs. Dip")
        axs[1,2].set_xlabel("Rake (Degrees)")
        axs[1,2].set_ylabel("Dip (Degrees)")
        if true_sol != []:
            axs[1,2].scatter(true_sol[2], true_sol[1], c='red', marker='o', s=100)
        
        axs[2,0].scatter(df["Strike1"]._append(df["Strike2"]),
                        df["Rake1"]._append(df["Rake2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[2,0].set_title("Strike vs. Rake")
        axs[2,0].set_xlabel("Strike (Degrees)")
        axs[2,0].set_ylabel("Rake (Degrees)")
        if true_sol != []:
            axs[2,0].scatter(true_sol[0], true_sol[2], c='red', marker='o', s=100)
        
        axs[2,1].scatter(df["Dip1"]._append(df["Dip2"]),
                        df["Rake1"]._append(df["Rake2"]),
                        c=df[weight]._append(df[weight]), cmap="YlGnBu")
        axs[2,1].set_title("Dip vs. Rake")
        axs[2,1].set_xlabel("Dip (Degrees)")
        axs[2,1].set_ylabel("Rake (Degrees)")
        if true_sol != []:
            axs[2,1].scatter(true_sol[1], true_sol[2], c='red', marker='o', s=100)
        
        axs[2,2].hist(df["Rake1"]._append(df["Rake2"]), bins=np.linspace(-180,180,bins))
        if true_sol != []:
            axs[2,2].axvline(x=true_sol[2][0], color='green', label='Rake1')
            axs[2,2].axvline(x=true_sol[2][1], color='red', label='Rake2')
            axs[2,2].legend()
        axs[2,2].set_title("Rake")
        axs[2,2].set_xlabel("Degrees")
        axs[2,2].set_ylabel("Frequency")
        
        plt.show()
    
    elif type == "amp":
        
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        if weight == "OldWeight":
            plt.suptitle("Weighted Amplitude Scatter Plots (Old Method)", fontsize=20)
        else:
            plt.suptitle("Weighted Amplitude Scatter Plots (New Method)", fontsize=20)
        
        axs[0,0].hist(df["AP"], bins=bins)
        axs[0,0].set_title("AP")
        axs[0,0].set_xlabel("AP (m)")
        axs[0,0].set_ylabel("Frequency")
        
        axs[0,1].scatter(df["ASV"], df["AP"], c=df[weight], cmap="YlGnBu")
        axs[0,1].set_title("ASV vs. AP")
        axs[0,1].set_xlabel("ASV (m)")
        axs[0,1].set_ylabel("AP (m)")
        
        axs[0,2].scatter(df["ASH"], df["AP"], c=df[weight], cmap="YlGnBu")
        axs[0,2].set_title("ASH vs. AP")
        axs[0,2].set_xlabel("ASH (m)")
        axs[0,2].set_ylabel("AP (m)")
               
        axs[1,0].scatter(df["AP"], df["ASV"], c=df[weight], cmap="YlGnBu")
        axs[1,0].set_title("AP vs. ASV")
        axs[1,0].set_xlabel("AP (m)")
        axs[1,0].set_ylabel("ASV (m)")
        
        axs[1,1].hist(df["ASV"], bins=bins)
        axs[1,1].set_title("ASV")
        axs[1,1].set_xlabel("ASV (m)")
        axs[1,1].set_ylabel("Frequency")
        
        axs[1,2].scatter(df["ASH"], df["ASV"], c=df[weight], cmap="YlGnBu")
        axs[1,2].set_title("ASH vs. ASV")
        axs[1,2].set_xlabel("ASH (m)")
        axs[1,2].set_ylabel("ASV (m)")
        
        axs[2,0].scatter(df["AP"], df["ASH"], c=df[weight], cmap="YlGnBu")
        axs[2,0].set_title("AP vs. ASH")
        axs[2,0].set_xlabel("AP (m)")
        axs[2,0].set_ylabel("ASH (m)")
        
        axs[2,1].scatter(df["ASV"], df["ASH"], c=df[weight], cmap="YlGnBu")
        axs[2,1].set_title("ASV vs. ASH")
        axs[2,1].set_xlabel("ASV (m)")
        axs[2,1].set_ylabel("ASH (m)")
        
        axs[2,2].hist(df["ASH"], bins=bins)
        axs[2,2].set_title("ASH")
        axs[2,2].set_xlabel("ASH (m)")
        axs[2,2].set_ylabel("Frequency")
        
        plt.show()
        
    else:
        print("Invalid type")
        
def plot_cross(ax, t: list, p: list, weight: float = 1, t_color: str = "black", p_color: str = "red"):
    """
    Plot a t-p cross in 3D

    Args:
        ax: axes to plot on
        t (list): [x,y,z] coordinates of t axis
        p (list): [x,y,z] coordinates of p axis
        weight (float, optional): weight of line. Defaults to 1.
    """
    t_prime = other_sphere_point(t)
    p_prime = other_sphere_point(p)
    ax.plot([t[0], t_prime[0]], [t[1], t_prime[1]], [t[2], t_prime[2]], color=t_color, alpha=weight)
    ax.plot([p[0], p_prime[0]], [p[1], p_prime[1]], [p[2], p_prime[2]], color=p_color, alpha=weight)

def plot_crosses(df: pd.DataFrame, weight: str):
    """
    Plot t-p crosses for each solution in a dataframe

    Args:
        df (pd.DataFrame): dataframe of solutions
        weight (str): column name of weight (old or new)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_weight = df[weight].min()
    max_weight = df[weight].max()
    range_weight = max_weight - min_weight
    
    for i in range(len(df)):
        t = np.array([df.iloc[i]["tx"], df.iloc[i]["ty"], df.iloc[i]["tz"]])
        p = np.array([df.iloc[i]["px"], df.iloc[i]["py"], df.iloc[i]["pz"]])
        plot_cross(ax, t, p, (df.iloc[i][weight]-min_weight)/(10*range_weight))
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ax.set_title(f"{weight} t-p Crosses")
    
    plt.show()
    
# Decide whether to save, show or save or return
def plot_beachball_set(df: pd.DataFrame, weight: str):
    """
    Plot beachballs for each solution in a dataframe

    Args:
        df (pd.DataFrame): dataframe of solutions
        weight (str): column name of weight (old or new)
    """
    fig, ax = plt.subplots()
    min_weight = df[weight].min()
    max_weight = df[weight].max()
    range_weight = max_weight - min_weight
    
    for i in range(len(df)):
        strike = df.iloc[i]["Strike1"]
        dip = df.iloc[i]["Dip1"]
        rake = df.iloc[i]["Rake1"]
        collection = beach([strike, dip, rake],
                           alpha=(df.iloc[i][weight] - min_weight)/(100*range_weight),
                           edgecolor=None)
        ax.add_collection(collection)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal')
    
    plt.axis('off')
    plt.show()
    
def aggregate_sdr(df: pd.DataFrame, weight: str):
    """
    Calculate aggregate strike, dip, and rake from a dataframe of solutions

    Args:
        df (pd.DataFrame): dataframe of solutions
        weight (str): column name of weight (old or new)

    Returns:
        list: [strike, dip, rake] of aggregate solution
    """
    strike1, strike2 = 0, 0
    dip1, dip2 = 0, 0
    rake1, rake2 = 0, 0
    for i in range(len(df)):
        strike1 += df.iloc[i]["Strike1"]*df.iloc[i][weight]
        dip1 += df.iloc[i]["Dip1"]*df.iloc[i][weight]
        rake1 += df.iloc[i]["Rake1"]*df.iloc[i][weight]
        strike2 += df.iloc[i]["Strike2"]*df.iloc[i][weight]
        dip2 += df.iloc[i]["Dip2"]*df.iloc[i][weight]
        rake2 += df.iloc[i]["Rake2"]*df.iloc[i][weight]
    strike1 /= df[weight].sum()
    dip1 /= df[weight].sum()
    rake1 /= df[weight].sum()
    strike2 /= df[weight].sum()
    dip2 /= df[weight].sum()
    rake2 /= df[weight].sum()
    return np.array([strike1, dip1, rake1]), np.array([strike2, dip2, rake2])

# Put beachballs on visuals
# Plot aggregate beachballs
# Calculate misfit angle of aggregate beachballs
# Add vertical lines in my histograms
# Generate datasets without alpha

"""
ENTER MONDAY WITH A PLAN FOR THE STATISTICS - yes
GET AMPLITUDE SPACE VISUALIZATIONS
CREATE A FUNCTION FOR AMPLITUDE SPACE VISUALIZATIONS
INCORPORATE B3_OVER_A3
WHAT'S THAT BUG??? --> not a bug
"""

"""
So it's definitely not tp2sdr
Who knows, it may not even be a bug?
No, that's a lie, it's definitely a bug

It's probably the rotation matrix
It's definitely not the rotation matrix

Turns out it's not a bug
"""

"""
Use a uniform scale for histograms!!!
Not a bug, use analysis of the method's geometry
Method modifications/mathematical advances in report
"""

"""
Advances:
Use smaller ellipsoids for more definitive visualizations
Plot the vectors of amplitudes
"""

"""
there's something about the shape of the ellipsoid...
test my_tp2sdr rigorously against tp2sdr

Forward problem: scan through entire space
Plot same mechanism over
az = [0,360] from compass north
takeoff = [0,180] from south pole
turn distance into takeoff angles, scan distances
revise the beachball code to accurately represent the beachball
"""
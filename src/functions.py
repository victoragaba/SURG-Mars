'''
Name: Victor Agaba

Date: 4th November 2024

The goal of this module is to provide all helper functions needed for the
seismic inversion project.

'''


import numpy as np
from numpy import linalg
import pandas as pd
from matplotlib import pyplot as plt
from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import beach


eps = 1e-10; halfpi = np.pi/2; twopi = 2*np.pi
i_hat = np.array([1,0,0]); j_hat = np.array([0,1,0]); k_hat = np.array([0,0,1])


def unit_vec(vector: list) -> list:
    ''' Returns the unit vector of a vector. '''
    return vector / linalg.norm(vector)


def boundary_points(Ao: list, Uo: list, As: list = [], chi2: float = 1) -> tuple:
    '''
    Returns the boundary points of the model.

    Args:
        Ao (list): Vector of observed amplitudes.
        Uo (list): Vector of measured uncertainties.
        As (list, optional): Vector of simulated amplitudes. Defaults to [].
        chi2 (float, optional): _description_. Defaults to 1.
    '''
    # get a simulation for spherical model
    if len(As) == 0:
        As = i_hat
        if abs(Ao @ j_hat) < eps: As = As + j_hat 
    
    # adjusted covariance matrix
    assert all(Uo > 0), "All uncertainties must be positive."
    Sigma_tilde = np.diag(Uo)/chi2
    Sigma_tilde_inv = linalg.inv(Sigma_tilde)
    
    # constants
    c1 = Ao @ Sigma_tilde_inv @ Ao
    c2 = Ao @ Sigma_tilde_inv @ As
    c3 = As @ Sigma_tilde_inv @ As
    
    # conditions
    assert c1 >= 1, "COnfidence ellipsoid is too big."
    assert c1*c3 > c2**2, "Cauchy-Schwarz inequality is not satisfied."
    
    # boundary points
    left = (1 - 1/c1)*Ao
    right = (1/c1)*np.sqrt((c1-1)/(c1*c3 - c2**2))*(c1*As - c2*Ao)
    Ab_low, Ab_high = left - right, left + right
    
    return Ab_low, Ab_high
    

def get_epsilon(Ao: list, bounds: tuple) -> float:
    '''
    Returns the max angle between the observed and simulated amplitudes.
    Output is in RADIANS.

    Args:
        Ao (list): Vector of observed amplitudes.
        bounds (tuple): Tuple of boundary points.
    '''
    Ab_low, Ab_high = bounds
    angle_low = np.arccos(unit_vec(Ao) @ unit_vec(Ab_low))
    angle_high = np.arccos(unit_vec(Ao) @ unit_vec(Ab_high))
    
    return max(angle_low, angle_high)


def starting_direc(point: list, direc: list) -> list:
    '''
    Find what vector is perpendicular to point vector and lies in the
    plane that contains point and direc.

    Args:
        point (list): [x,y,z] on a unit sphere
        direc (list): direction to match, unit length
    '''
    
    proj = (np.dot(direc, point)/np.dot(point, point))
    rem = np.array(direc) - proj*np.array(point)
    return rem/linalg.norm(rem) if linalg.norm(rem) > eps else j_hat


def rect2pol(rect: list) -> list:
    '''
    Converts from rectangular to polar/spherical coordinates.

    Args:
        rect (list): rectangular coordinates
    '''
    
    if len(rect) == 2:
        x, y = rect
        r = linalg.norm(rect)
        theta = np.arctan2(y, x) % twopi
        return np.array([r, theta])
    
    x, y, z = rect
    rho = linalg.norm(rect)
    theta = np.arctan2(y, x) % twopi
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    return np.array([rho, theta, phi])


def pol2rect(pol: list) -> list:
    '''
    Converts from polar/spherical to rectangular coordinates.

    Args:
        pol (list[int]): polar/spherical coordinates
    '''
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


def angle2bearing(angle: float) -> float:
    '''
    Converts angle (unit circle) to bearing.
    Angles in RADIANS.

    Args:
        angle (float): angle mod 2*pi in RADIANS
    '''
    angle = twopi - angle
    bearing = (angle + halfpi) % twopi
    
    return bearing


def bearing2angle(bearing: float) -> float:
    '''
    Converts bearing to angle (unit circle).
    Angles in RADIANS.

    Args:
        bearing (float): bearing in RADIANS, N = y axis
    '''
    bearing = twopi - bearing
    angle = (bearing + halfpi) % twopi
    
    return angle


def rotate_vec(vec: list, axis: list, theta: float) -> list:
    """
    Rotate vec around axis by theta RADIANS, counterclockwise.
    Axis must be a unit vector.
    """
    rotated_vec = vec*np.cos(theta) + (np.cross(axis,vec))*np.sin(theta) + \
        axis*(np.dot(axis,vec))*(1-np.cos(theta))
    
    return rotated_vec


def tp2sdr(t: list, p: list, deg: bool = False) -> tuple:
    '''
    Converts from T and P axes to strike, dip, and rake of both fault planes
    Inputs are numpy arrays, assumes unit vectors
    Uses right-hand rule for coordinate system

    Args:
        t (list): T axis
        p (list): P axis
        deg (bool, optional): output in degrees if True
    '''
    # get the normals
    n1 = t + p
    n2 = t - p
    
    # restrict to upper hemisphere
    if n1[2] < 0: n1 *= -1
    if n2[2] < 0: n2 *= -1
    
    # get spherical coordinates, dip is restricted phi
    _, theta1, dip1 = rect2pol(n1)
    _, theta2, dip2 = rect2pol(n2)
    
    # strike is theta measured in reverse
    strike1 = (twopi - theta1) % twopi
    strike2 = (twopi - theta2) % twopi
    
    # base directions for rakes, null for exceptions
    base1 = pol2rect(np.array([1, bearing2angle(strike1), halfpi]))
    base2 = pol2rect(np.array([1, bearing2angle(strike2), halfpi]))
    null = np.cross(t,p)
    if null[2] < 0: null *= -1
    done1, done2 = False, False
    
    # account for edge cases on perfectly vertical dips
    if n1[2] < eps and n2[2] < eps: # double vertical dip
        check1 = rotate_vec(base1, null, np.pi/4)
        if abs(np.dot(check1, t)) < eps: rake1 = 0.
        else: rake1 = np.pi
        check2 = rotate_vec(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done1, done2 = True, True
    elif n1[2] < eps: # single vertical dip
        check2 = rotate_vec(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done2 = True
    elif n2[2] < eps: # single vertical dip
        check1 = rotate_vec(base1, null, np.pi/4)
        if abs(np.dot(check1, t)) < eps: rake1 = 0.
        else: rake1 = np.pi
        done1 = True
    
    # check if the restricted normals house a t or p axis
    if abs(np.dot(n1 + n2, t)) < eps: # houses a p axis, negative slip
        slip1, slip2 = -n2, -n1
    else: # houses a t axis, positive slip
        slip1, slip2 = n2, n1

    # get rakes from  base and slip vectors
    if not done1:
        rake1 = np.arccos(np.dot(slip1, base1)/linalg.norm(slip1))
        rake1 *= np.sign(slip1[2])
    if not done2:
        rake2 = np.arccos(np.dot(slip2, base2)/linalg.norm(slip2))
        rake2 *= np.sign(slip2[2])
    
    if deg:
        strike1, strike2 = np.rad2deg(strike1), np.rad2deg(strike2)
        dip1, dip2 = np.rad2deg(dip1), np.rad2deg(dip2)
        rake1, rake2 = np.rad2deg(rake1), np.rad2deg(rake2)
    
    return (np.array([strike1, dip1, rake1]), np.array([strike2, dip2, rake2]))


def sdr2tp(sdr: list, deg: bool = False) -> tuple:
    """
    Converts from strike, dip, and rake to T and P axes.

    Args:
        sdr (list): strike, dip, and rake
        deg (bool, optional): input in degrees if True

    Returns:
        tuple: (t, p) as unit vectors in x, y, z
    """
    if deg: sdr = np.deg2rad(sdr)
    
    # first normal from strike and dip
    strike, dip, rake = sdr
    
    n1_theta = bearing2angle(strike + halfpi)
    n1_phi = dip
    n1 = pol2rect(np.array([1, n1_theta, n1_phi]))

    n2_theta = bearing2angle(strike)
    n2_init = pol2rect(np.array([1, n2_theta, halfpi]))
    n2 = rotate_vec(n2_init, n1, rake)
    
    # get T and P axes
    t = (n1 + n2)/linalg.norm(n1 + n2)
    p = (n1 - n2)/linalg.norm(n1 - n2)
    
    # restrict to upper hemisphere
    if t[2] < 0: t *= -1
    if p[2] < 0: p *= -1
    
    return t, p


def bound(params: list) -> list:
    '''
    Put params in the range [0, 2pi], [0, pi/2], [-pi, pi].
    Intended to correct optimal solution from steepest descent algorithm.
    
    Args:
        params (list): [strike, dip, rake] in RADIANS
    '''
    
    turns = [halfpi, np.pi, twopi]
    
    # leave input unchanged
    out = params.copy()
    
    # driven by dip: case for every quadrant
    overturned = out[1] % turns[2] > turns[1]
    out[1] = out[1] % turns[1]
    
    # strike-dip relationship
    if out[1] > turns[0]:
        out[1] = turns[1] - out[1]
        out[0] += turns[1]
    out[0] = out[0] % turns[2]
    
    # bound the rake
    if overturned: out[2] += turns[1]
    out[2] = out[2] % turns[2]
    if out[2] > turns[1]:
        out[2] -= turns[2]
    
    return out


def systematic_hemisphere_samples(step_size: float) -> list:
    """
    Systematically samples n samples evenly spaced on an upper hemisphere surface.

    Args:
        step_size (float): angular distance between samples in DEGREES

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


def systematic_params(dd: float) -> list:
    '''
    Returns n systematically sampled parameters for the inversion.
    Sampled from tp space for better coverage.
    Output is in RADIANS.
    
    Args:
        dd (float): angular distance between samples in DEGREES
    '''
    params = []
    Ts = systematic_hemisphere_samples(dd)
    P_rotations = np.arange(0, np.pi, np.deg2rad(dd))
    for T in Ts:
        P_start = starting_direc(T, i_hat + j_hat + k_hat)
        for theta in P_rotations:
            P = rotate_vec(P_start, T, theta)
            param = tp2sdr(T, P)[0]
            params.append(param)
    
    return np.array(params)


def orthogonalize(v: list, u: list) -> list:
    '''
    Orthogonalize vector v to u.
    
    Args:
        v (list): vector to orthogonalize
        u (list): vector to orthogonalize to
    '''
    assert linalg.norm(u) > eps, "u must be a non-zero vector."
    assert linalg.norm(v) > eps, "v must be a non-zero vector."
    if np.abs(1 - np.abs(unit_vec(v) @ unit_vec(u))) < eps:
        return np.zeros(3)
    return unit_vec(v - (v @ u)/(u @ u)*u)


def random_hemisphere_samples(n: int) -> list:
    '''
    Select n random samples uniformly distributed on the surface of an
    upper hemisphere of unit radius

    Args:
        n (int): # samples

    Returns:
        list: array of [x,y,z] coordinates
    '''
    samples = []

    for i in range(n):
        # generate from normal distribution
        p = np.random.normal(size=3)
        while linalg.norm(p) < eps or all(p[:2]) == 0:
            p = np.random.normal(size=3)
        p = unit_vec(p)
        
        # restrict to upper hemisphere
        p[2] = np.abs(p[2])
        
        samples.append(p)
    
    return samples


def random_params(n:int) -> list:
    '''
    Returns n random parameters for the inversion.
    Sampled uniformly from TP space.
    Output is in RADIANS.
    '''
    Ts = random_hemisphere_samples(n)
    raw_Ps = random_hemisphere_samples(n)
    Ps = []
    for i in range(n):
        Ptry = orthogonalize(raw_Ps[i], Ts[i])
        while linalg.norm(Ptry) < eps:
            print(f"Ptry: {Ptry}")
            raw_P = random_hemisphere_samples(1)[0]
            Ptry = orthogonalize(raw_P, Ts[i])
        Ps.append(Ptry)
    params = [tp2sdr(Ts[i], Ps[i])[0] for i in range(n)]
    
    return np.array(params)


def plot_beachball_set(params: list) -> None:
    '''
    Returns a plot of beachballs for a set of focal mechanisms.
    
    Args:
        params (list): list of SDR parameters
    '''
    pass # Omkar


def kagan_angle(sdr1: list, sdr2: list) -> float:
    '''
    Returns the Kagan angle between two focal mechanisms given their SDR parameters.
    Output is in RADIANS.
    
    Args:
        sdr1 (list): first mechanism's SDR parameters in RADIANS
        sdr2 (list): second mechanism's SDR parameters in RADIANS
    '''
    pass # Nseko


def filter_axes(tp_axes: list) -> list:
    '''
    Filters out the T and P axes to get one side of the cone.
    
    Args:
        tps (list): list of tp axes
    '''
    # pick a random tp pair
    t_ref, p_ref = tp_axes[0]
    
    # output list
    tp_filtered = np.zeros((len(tp_axes), 2, 3))
    
    # use sign of dot product to determine side
    for i, (t, p) in enumerate(tp_axes):
        if np.dot(t, t_ref) < 0: t *= -1
        if np.dot(p, p_ref) < 0: p *= -1
        tp_filtered[i] = np.array([t, p])
    
    return tp_filtered


def regression_plane(data: list) -> tuple:
    '''
    Returns the best fit plane for a set of data points.
    
    Args:
        data (list): list of data points
    '''
    centroid = np.mean(data, axis=0)
    A = np.c_[data, np.ones(data.shape[0])]
    print(f'Rank of A: {linalg.matrix_rank(A)}')
    # TODO: why is this not working?
    normal = np.lin
    normal = normal[:-1]
    print(f"Centroid: {centroid}, Normal: {normal}")
    
    return centroid, normal

def regression_axes(tp_axes: list) -> tuple:
    '''
    Returns the best fit plane for a set of TP axis endpoints.
    Assumes they are alreaady filtered.
    Uses the narrower cone to determine the plane.
    
    Args:
        tp_axes (list): list of tp axes
    '''
    Ts = np.array([tp[0] for tp in tp_axes])
    Ps = np.array([tp[1] for tp in tp_axes])
    
    T_centroid, T_normal = regression_plane(Ts)
    P_centroid, P_normal = regression_plane(Ps)
    
    # normalize all vecs
    T_try, T_normal = unit_vec(T_centroid), unit_vec(T_normal)
    P_try, P_normal = unit_vec(P_centroid), unit_vec(P_normal)
    
    # return Centroid and normal whose dot product is largest
    if T_try @ T_normal > P_try @ P_normal:
        return T_centroid, T_normal
    else:
        return P_centroid, P_normal

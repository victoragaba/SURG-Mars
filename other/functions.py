'''
Name: Victor Agaba

Date: 23rd January 2023

The goal of this file is to implement functions necessary for
the overall project.

Functions:
--fill in later--

Notes:
starting direc has been modified to restrict to the upper hemisphere
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import collections
from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import beach

eps = 1e-10; halfpi = np.pi/2; twopi = 2*np.pi
i_hat = np.array([1,0,0]); j_hat = np.array([0,1,0]); k_hat = np.array([0,0,1])
np.random.seed(2024)


def rigid_hemisphere_samples(step_size: float, angles: bool = False) -> list:
    """
    Systematically select n samples evenly spaced on the surface of an upper hemisphere.

    Args:
        step_size (float): angular distance between samples in degrees
        angles (bool, optional): return in angles if True (1, theta, phi)

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
        
    return samples if not angles else [rect2pol(s) for s in samples]


def Rpattern(fault, azimuth, takeoff_angles, alpha_beta: list = []):
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
    P as measured on L (~Z) component, SV measured on Q (~R) component, and SH measured on T
    component.
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


def rotate_vector(vec: list, axis: list, theta: float) -> list:
    """
    Rotate vec around axis by theta radians (right-hand rule)
    i.e. align right thumb with axis, fingers curl in rotation direction.
    Assumes axis is normalized.
    ANGLES IN RADIANS!!!
    """
    rotated_vec = vec*np.cos(theta) + \
        np.cross(axis, vec)*np.sin(theta) + \
        axis*np.dot(axis, vec)*(1 - np.cos(theta))
    
    return rotated_vec


def bearing2angle(bearing: float) -> float:
    """
    Convert bearing to angle (unit circle).

    Args:
        bearing (float): bearing in radians, N = y axis

    Returns:
        float: angle (unit circle) mod 2*pi in radians
    """
    bearing = twopi - bearing
    angle = (bearing + halfpi) % twopi
    
    return angle


def rect2pol(rect: list) -> list:
    """
    Convert from rectangular to polar/spherical coordinates.

    Args:
        rect (list): rectangular coordinates

    Returns:
        list: spherical coordinates
    """
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


def pol2rect(pol: list) -> list:
    """
    Convert from polar/spherical to rectangular coordinates.

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


def tp2sdr(t: list, p: list, deg: bool = False) -> tuple:
    """
    Converts from T and P axes to strike, dip, and rake of both fault planes.
    Inputs are numpy arrays, assumes unit vectors.
    Uses right-hand rule for coordinate system.

    Args:
        t (list): T axis unit vector
        p (list): P axis unit vector
        deg (bool, optional): output in degrees if True

    Returns:
        tuple: (sdr1, sdr2)
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
        check1 = rotate_vector(base1, null, np.pi/4)
        if abs(np.dot(check1, t)) < eps: rake1 = 0.
        else: rake1 = np.pi
        check2 = rotate_vector(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done1, done2 = True, True
    elif n1[2] < eps: # single vertical dip
        check2 = rotate_vector(base2, null, np.pi/4)
        if abs(np.dot(check2, t)) < eps: rake2 = 0.
        else: rake2 = np.pi
        done2 = True
    elif n2[2] < eps: # single vertical dip
        check1 = rotate_vector(base1, null, np.pi/4)
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
    
    return np.array([strike1, dip1, rake1]), np.array([strike2, dip2, rake2])


def sdr2tp(sdr: list, deg: bool = False) -> tuple:
    """
    Converts from strike, dip, and rake to T and P axes.
    ANGLES IN RADIANS!!!

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
    n2 = rotate_vector(n2_init, n1, rake)
    
    # get T and P axes
    t = (n1 + n2)/np.linalg.norm(n1 + n2)
    p = (n1 - n2)/np.linalg.norm(n1 - n2)
    
    # restrict to upper hemisphere
    if t[2] < 0: t *= -1
    if p[2] < 0: p *= -1
    
    return t, p


def starting_direc(point: list, normal: list, backup: list = i_hat) -> list:
    """
    Find what vector is perpendicular to point and norm.
    Useful setup for locking every orientation in one hemisphere.
    Enforces right-hand rule.

    Args:
        point (list): [x,y,z] on a unit sphere
        normal (list): [x,y,z] of the normal vector for embedding plane
        backup (list, optional): backup vector if norm is parallel to point

    Returns:
        list: [x,y,z] of the starting direction
    """
    if 1 - abs(np.dot(normal, point)) < eps: point = backup
    start = np.cross(normal, point) # enforces right-hand rule
    start = start/np.linalg.norm(start)
    
    return start


def cossim(u: np.array, v: np.array) -> float:
    '''
    Calculates the cosine similarity between two vectors.
    '''
    return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))


def bound_sdr(sdr: list, deg: bool = False, dont: bool = False) -> list:
    
    
    
    # THIS NEEDS FIXING
    
    # Rotate the physical beachball to make sure it works
    
    
    
    '''
    Put sdr in the range [0, 360], [0, 90], [-180, 180].
    deg: input in degrees if True.
    Returns sdr in same units as input.
    dont: if True, don't bound (for plotting iterates).
    '''
    if dont: return sdr
    
    turns = [halfpi, np.pi, twopi]
    if deg: turns = [90, 180, 360]
    
    out = sdr.copy()
    
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


def grid_search(dd, Ao, azimuth, takeoff_angles, alpha, beta) -> tuple:
    '''
    Implements grid search as an inversion method.
    
    Args:
        dd: step size for grid search
        Ao: observed amplitudes
        azimuth: azimuth with which ray path leaves source (clockwise from N)
        takeoff_angles: angle between P ray path & vertical in the source model layer
        alpha: ratio of S-wave velocity to P-wave velocity
        beta: ratio of SH-wave velocity to P-wave velocity
        
    Returns:
        best_sdr: best strike, dip, and rake
    '''
    
    # look for sdr that minimizes the misfit function
    best_sdr = np.zeros(3)
    best_misfit = np.inf
    
    # grid search in tp space, convert to sdr: equivalent
    Ts = rigid_hemisphere_samples(dd)
    P_rotations = np.arange(0, np.pi, np.deg2rad(dd))
    counter = 0
    for T in Ts:
        P_start = starting_direc(T, i_hat + j_hat + k_hat)
        for theta in P_rotations:
            P = rotate_vector(P_start, T, theta)
            sdr = tp2sdr(T, P, True)[0]
            
            # do forward model
            As = Rpattern(sdr, azimuth, takeoff_angles, [alpha, beta])
            
            # look at negative of cosine similarity
            misfit = -cossim(Ao, As)
            
            # update best_sdr if necessary
            if misfit < best_misfit:
                best_sdr = sdr
                best_misfit = misfit
            
            # update counter
            counter += 1
    
    print(f'Solution found in {counter} iterations.')
    
    return best_sdr, best_misfit


def hierarchical_grid_search(num_div, zoom_times, Ao, azimuth, takeoff_angles, alpha, beta) -> tuple:
    '''
    Implements hierarchical grid search as an inversion method.
    num_div: number of divisions in the first grid search
    zoom_times: number of times to zoom in on the best sdr
    '''
    
    # do a grid search in sdr space, make a of num_div**3 points
    # restrict to a hemisphere
    strike_range = np.linspace(0, 360, num_div)
    dip_range = np.linspace(0, 90, num_div)
    rake_range = np.linspace(-90, 90, num_div)
    
    # take note of windows
    strike_window = 2*np.pi / num_div
    dip_window = np.pi / (2*num_div)
    rake_window = np.pi / num_div
    
    counter = 0
    
    def mini_grid_search(counter):
        '''
        Implements a grid search in sdr space.
        '''
        # look for sdr that minimizes the misfit function
        best_sdr = np.zeros(3)
        best_misfit = np.inf
        
        # grid search in sdr space
        for strike in strike_range:
            for dip in dip_range:
                for rake in rake_range:
                    sdr = np.array([strike, dip, rake])
                    
                    # do forward model
                    As = Rpattern(sdr, azimuth, takeoff_angles, [alpha, beta])
                    
                    # look at negative of cosine similarity
                    misfit = -cossim(Ao, As)
                    
                    # update best_sdr if necessary
                    if misfit < best_misfit:
                        best_sdr = sdr
                        best_misfit = misfit
                    
                    counter += 1
        
        return best_sdr, best_misfit, counter
    
    while zoom_times > 0:
        # perform a grid search
        best_sdr, best_misfit, counter = mini_grid_search(counter)
        
        # zoom in on the best sdr
        strike_range = np.linspace(best_sdr[0] - strike_window/2, best_sdr[0] + strike_window/2, num_div)
        dip_range = np.linspace(best_sdr[1] - dip_window/2, best_sdr[1] + dip_window/2, num_div)
        rake_range = np.linspace(best_sdr[2] - rake_window/2, best_sdr[2] + rake_window/2, num_div)
        
        # update windows
        strike_window /= num_div
        dip_window /= num_div
        rake_window /= num_div
        zoom_times -= 1
    
    print(f'Solution found in {counter} iterations.')
    
    return best_sdr, best_misfit


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

eps = 1e-8; halfpi = np.pi/2; twopi = 2*np.pi
i_hat = np.array([1,0,0]); j_hat = np.array([0,1,0]); k_hat = np.array([0,0,1])

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
    input strile-dip pair

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
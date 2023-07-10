import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
MY NOTES:

Find the map between strike-dip pairs and their equivalents in their 'lower halves'
(Do they have lower halves? Is the entire space covered?)
(Think 91deg strike vs -89deg, or neg dip)

Dip is always positive acute for 1-1 mapping
Strike has a range [0, 2*pi)

Work on sphere_to_sd and sd_to_sphere (test, fix and test)
"""

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
        p[2] = abs(p[2]) # only upper hemisphere
        samples.append(p)
    
    return samples

def space_evenly(n: int, bounds: list) -> list:
    """
    Divide bounds into n evenly spaced points

    Args:
        n (int): # divisions
        bounds (list[float]): array bounds [start, end]
        
    Returns:
        list: n-size array of floats
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
        return [x,y]
    
    phi = pol[2]
    
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    
    return [x,y,z]

def rect2pol(rect: list) -> list:
    """
    Convert from rectangular to polar/spherical coordinates

    Args:
        rect (list): rectangular coordinates

    Returns:
        list: spherical coordinates
    """
    r = np.linalg.norm(rect)
    assert rect[:2] != [0,0]
    if rect[0] != 0:
        theta_acute = np.arctan(rect[1]/rect[0])
    else:
        theta_acute = np.pi
    
    if np.sign(rect[0]) > 0:
        if np.sign(rect[1]) >= 0:
            theta = theta_acute
        elif np.sign(rect[1]) < 0:
            theta = 2*np.pi + theta_acute
    elif np.sign(rect[0]) < 0:
        theta = np.pi + theta_acute
    else:
        if np.sign(rect[1]) > 0:
            theta = np.pi/2
        else:
            theta = 3*(np.pi/2)
    
    if len(rect) == 3:
        phi = np.arccos(rect[2]/r)
        return [r, theta, phi]
    
    return [r, theta]

def rad2deg(rad: float) -> float:
    """
    Convert angle from radians to degrees

    Args:
        rad (float): angle in radians

    Returns:
        float: angle in degrees
    """
    return rad/np.pi*180

def deg2rad(deg: float) -> float:
    """
    Convert angle from degrees to radians

    Args:
        deg (float): angle in degrees

    Returns:
        float: angle in radians
    """
    return deg/180*np.pi

def angle2bearing(angle: float) -> float:
    """
    Convert angle (unit circle) to bearing

    Args:
        angle (float): angle mod 2*pi in radians

    Returns:
        float: bearing in radians, N = y axis
    """
    angle = 2*np.pi - angle
    bearing = (angle + np.pi/2) % (2*np.pi)
    
    return bearing  

def bearing2angle(bearing: float) -> float:
    """
    Convert bearing to angle (unit circle)

    Args:
        bearing (float): bearing in radians, N = y axis

    Returns:
        float: angle (unit circle) mod 2*pi in radians
    """
    bearing = 2*np.pi - bearing
    angle = (bearing + np.pi/2) % (2*np.pi)
    
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
    if comp > np.pi/2: comp = np.pi - comp
    
    return np.pi/2 - comp
    
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
    if angle > np.pi/2: angle = np.pi - angle
    
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
    plane_angle = (spherical[1] + np.pi/2) % (2*np.pi)
    strike = angle2bearing(plane_angle)
    dip = spherical[2]
    
    return [strike, dip]

def sd_to_sphere(sd: list) -> list:
    """
    Find a point on the surface of an upper hemisphere corresponding to
    input strile-dip pair

    Args:
        sd (list): [strike, dip] pair

    Returns:
        list: numpy array of [x,y,z] coordinates
    """
    r = 1
    normal_bearing = (sd[0] + np.pi/2) % (2*np.pi)
    theta = bearing2angle(normal_bearing)
    phi = sd[1]
    
    return pol2rect([r, theta, phi])


if __name__ == '__main__':
    
    pass
    
    # print(rect2pol([0,-1,-1]))
    # print(rad2deg(np.arctan(np.tan(deg2rad(120)))))
    
    # point = [1,-1,np.sqrt(2)]
    # point /= np.linalg.norm(point)
    # print(point)
    # sd = sphere_to_sd(point)
    # print([rad2deg(rad) for rad in sd])
    # print(f"Point: {point}")
    # point2 = sd_to_sphere(sd)
    # print(f"Point2: {point2}")
    
    # test_sd = [deg2rad(45)]*2
    # print(test_sd)
    # print(sd_to_sphere(sphere_to_sd([0,0,1])))
    # point = [1,-1,1]
    # point /= np.linalg.norm(point)
    # print(point)
    
    # pol = rect2pol([-1,-1])
    # print(rad2deg(pol[1]))
    
    # rect = pol2rect([np.sqrt(2), deg2rad(315)])
    # print(rect)
    # print(rad2deg(angle2bearing(deg2rad(50))))
    # print(rad2deg(bearing2angle(deg2rad(-1))))
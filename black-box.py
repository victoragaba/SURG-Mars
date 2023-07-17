import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel

"""
MY NOTES:

Find the map between strike-dip pairs and their equivalents in their 'lower halves'
(Do they have lower halves? Is the entire space covered?)
(Think 91deg strike vs -89deg, or neg dip)

Trim this down to only the essentials, then use function-repo
"""

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
    
def Rpattern(fault, azimuth, takeoff_angles):
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

if __name__ == '__main__':
    
    # ## GENERATING SAMPLES
    # N = 500
    # t_samples = hemisphere_samples(N**2)
    # sdr1 = []
    # sdr2 = []
    
    # for t in t_samples:
    #     p_rotations = uniform_samples(N, [0, twopi])
    #     p_start = starting_direc(t, k_hat)
    #     for theta in p_rotations:
    #         p = rotate(p_start, t, theta)
    #         sdr1_emt, sdr2_emt = tp2sdr(coord_switch(t), coord_switch(p))
    #         sdr1.append(sdr1_emt)
    #         sdr2.append(sdr2_emt)
    
    # # Histograms and scatter
    # # subplots for sdr1 and sdr2
    # fig1 = plt.figure(figsize=(15,12))
    # plt.suptitle("SDR distribution for Fault Planes")
    
    # ylabels = ["Strike", "Dip", "Rake"]
    
    # for i in range(3):
    #     plt.subplot(3,2,2*i+1)
    #     plt.hist([np.rad2deg(emt[i]) for emt in sdr1], bins=50, density=True)
    #     if i == 0: plt.title("1st fault plane")
    #     plt.ylabel(f"{ylabels[i]}")
    # plt.xlabel("Angles (degrees)")
    
    # for i in range(3):
    #     plt.subplot(3,2,2*(i+1))
    #     plt.hist([np.rad2deg(emt[i]) for emt in sdr2], bins=50, density=True)
    #     if i == 0: plt.title("2nd fault plane")
    #     plt.ylabel(f"{ylabels[i]}")
        
    # plt.xlabel("Angles (degrees)")
    # plt.show()
    
    """
    Explain the sinusoidal dip distribution
    How does it influence grid search/predictions?
    """
    
    ## TESTING PREDICTIVE POWER
    # https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_events.html
    model = TauPyModel(model='ak135')
    model_name = 'ak135'
    
    hdepth = 15   # km   - assumed quake depth  (a parameter that affects 
    # take-off angles, given a distance between quake and station)
    
    b3_over_a3 = (3.4600/5.8000)**3    # ideally this S- over P-velocity 
    # ratio cubed should be read directly from the velocity model that 
    # is being used. My hope is Caio can provide this functionality. 
    # The P and S velocities for this ratio are the velocities at the depth of the quake.
    
    epdist = 10   # good estimate? - use wilbur3, ask caio for corroboration methods
    arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                         distance_in_degree=epdist, phase_list=['P', 'S'])
    """
    a.name, a.distance, a.time, a.ray_param, a.takeoff_angle, a.incident_angle
    """
    print([(a.name, a.takeoff_angle, a.incident_angle) for a in arrivals])
    
    takeoff_angles = [a.takeoff_angle for a in arrivals]
    t, p = np.array([1,0,0]), np.array([0,0,1])
    faults = tp2sdr(coord_switch(t), coord_switch(p))
    faults = [np.rad2deg(np.array(sdr)) for sdr in faults]
    azimuth = 100
    amps1 = Rpattern(faults[0], azimuth, takeoff_angles)
    print(amps1)
    amps2 = Rpattern(faults[1], azimuth, takeoff_angles)
    print(amps2)
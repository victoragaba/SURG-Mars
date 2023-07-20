import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel

"""
MY NOTES:

Trim this down to only the essentials, then use function-repo

My method is more computationally expensive

Reverse rigid sampling to fix the hole at the pole

Look into elliptical distributions
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

def azdp(v, units): # Suzan
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

def tp2sdr(t,p): # Suzan
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
    
def Rpattern(fault, azimuth, takeoff_angles): #Omkar
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

def get_epsilon(Ao: list, Uo: list) -> float:
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
    e = np.sqrt(np.sum([mag_perc(sig[i],Ao)**2 for i in range(len(Ao))]))
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


if __name__ == '__main__':
    
    ## GENERATING SAMPLES
    # N = 100
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
    
    hdepth = 15 # km - assumed quake depth
    
    b3_over_a3 = (3.4600/5.8000)**3 # Caio --part of the velocity model?
    # Is this specific to 15 km depth?
    
    epdist = 10   # good estimate? - use wilbur3, ask caio for corroboration methods
    arrivals = model.get_travel_times(source_depth_in_km=hdepth,
                         distance_in_degree=epdist, phase_list=['P', 'S'])
    """
    a.name, a.distance, a.time, a.ray_param, a.takeoff_angle, a.incident_angle
    """
    # print([(a.name, a.takeoff_angle, a.incident_angle) for a in arrivals])
    
    takeoff_angles = [a.takeoff_angle for a in arrivals]
    azimuth = 100
    t, p = np.array([1,0,0]), np.array([0,0,1]) # normal fault
    faults = tp2sdr(coord_switch(t), coord_switch(p))
    faults = [np.rad2deg(np.array(sdr)) for sdr in faults]
    
    # fake an observation of amplitude + uncertainty (normal fault in this case)
    scale = stats.uniform.rvs(0.5,10) # necessary?
    Ao = np.array([i*scale for i in Rpattern(faults[0], azimuth, takeoff_angles)])
    Ao[0] *= b3_over_a3 # Maddy's paper page 5, reread
    sigma = 0.1
    Uo = np.array([abs(Ao[i])*sigma for i in range(len(Ao))])
    print(f"Observation: {Ao}\nUncertainty: {Uo}")
    
    """
    I want to freeze randomly generated data for testing purposes
    """
    
    # Get epsilon (Maddy's method)
    old_epsilon = get_epsilon(Ao, Uo)
    
    # Grid search + my version of epsilon
    N = 10 # small for now
    
    # old_accepted1 = [] # accepted fault planes within 1 standard deviation
    # old_accepted2 = [] # accepted fault planes within 2 standard deviations
    # old_rejected = []  # rejected fault planes
    # accepted1 = []; accepted2 = []; rejected = []
    
    filtered_sdrs = [[] for i in range(6)] # old and new accepted/rejected, same order
    # format per inner list: [([strike, dip, rake], weight), ...]
    
    t_samples = hemisphere_samples(N**2)
    for t in t_samples:
        p_rotations = uniform_samples(N, [0, twopi])
        p_start = starting_direc(t, k_hat)
        for theta in p_rotations:
            p = rotate(p_start, t, theta)
            sdr = tp2sdr(coord_switch(t), coord_switch(p))[0]
            sdr = np.rad2deg(np.array(sdr))
            As = np.array(Rpattern(sdr, azimuth, takeoff_angles))
            As[0] *= b3_over_a3
            angle = np.arccos(np.dot(As, Ao)/(np.linalg.norm(As)*np.linalg.norm(Ao)))
            
            # Maddy's method
            old_weight = get_gaussian_weight(angle, old_epsilon)
            if old_weight > np.exp(-1/2): # 1 standard deviation
                filtered_sdrs[0].append((sdr, old_weight))
            elif old_weight > np.exp(-2): # 2 standard deviations
                filtered_sdrs[1].append((sdr, old_weight))
            else:
                filtered_sdrs[2].append((sdr, old_weight))
                
            # My method
            # Continue from get_ellipse_epsilon
            epsilon = get_ellipse_epsilon(Ao, Uo, As)
            weight = get_gaussian_weight(angle, epsilon)
            if weight > np.exp(-1/2): # 1 standard deviation
                filtered_sdrs[3].append((sdr, weight))
            elif weight > np.exp(-2): # 2 standard deviations
                filtered_sdrs[4].append((sdr, weight))
            else:
                filtered_sdrs[5].append((sdr, weight))
            
    # Plotting/visualization
    # Start with histograms of accepted/rejected fault planes
    # Doing all 6 separately
    
    
    # find a way to extract all strikes, dips and rakes into separate lists
    split_sdrs = dict()
    split_sdrs["Strikes"] = [[np.rad2deg(emt[0][0]) for emt in filt] for filt in filtered_sdrs]
    split_sdrs["Dips"] = [[np.rad2deg(emt[0][1]) for emt in filt] for filt in filtered_sdrs]
    split_sdrs["Rakes"] = [[np.rad2deg(emt[0][2]) for emt in filt] for filt in filtered_sdrs]
    titles = ["Strikes", "Dips", "Rakes"]
    y_axes = ["Accepted (1 std)", "Accepted (2 std)", "Rejected (2 std)"]
    
    for i in range(3): # strikes, dips and rakes
        # main figures
        fig = plt.figure(figsize=(15,12))
        plt.suptitle(titles[i])
        
        for j in range(3): # (old + new) accepted1, accepted2, rejected
            # old
            plt.subplot(3,2,2*j+1) # subplots are 1-indexed
            plt.hist(split_sdrs[titles[i]][j], bins=50, density=True)
            if j == 0: plt.title("Old")
            plt.ylabel(y_axes[j])
            # new
            plt.subplot(3,2,2*(j+1))
            plt.hist(split_sdrs[titles[i]][j+3], bins=50, density=True)
            if j == 0: plt.title("New")
            plt.ylabel(y_axes[j])
    
        plt.show()
    
    """
    Draw out the data structures for debugging
    """
    
    # Stacked (2 standard deviations)
    
    
    # Scatter (3D sdr space)
    
    # Aggregate sdr if it's unimodal
    # How to locate modes?
    
    """
    First I try Gaussian weights
    Then trigonometric weights
    Then I try to combine the two
    """
    
    """
    Plot histograms of accepted/rejected fault planes
        stacked histogram
        effect of including more standard deviations?
    Plot beachballs for accepted fault planes
    How is a best fitting solution chosen? (Maddy's paper)
    Look into regression methods -- are they necessary?
    How to visualize weights?
        scatter plot
        something with a color density
    """
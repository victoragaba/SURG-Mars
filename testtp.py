import numpy as np

eps = 1e-8
halfpi = 0.5*np.pi

def azdp(v, units):
    """
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


def tp2sdr(t,p):
    """
    IN: vectors representing T and P axes (in r-theta-phi (normal mode) coordinates)
        reminder: r (up), theta (south), and phi (east)
    OUT: strike, dip, and rake for both fault planes (double couple assumed)
    """
    pole1 = (t+p)/np.sqrt(2.) 
    pole2 = (t-p)/np.sqrt(2.)
    azt,dpt = azdp(t,'deg')
    azp,dpp = azdp(p,'deg')
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

    return (st1,dip1,rake1), (st2,dip2,rake2)  # in radians


p1,p2 = tp2sdr(np.array([0,0,1]),np.array([1,0,0]))
p1d = np.degrees(np.array(p1))
p2d = np.degrees(np.array(p2))
print(p1d)
print(p2d)

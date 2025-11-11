
# functions written by Suzan van der Lee (SL) @1993 - 2022
#
# to compute focal shpere parameters from Moment Tensor and v.v.
#
# T-axis means tension at the source and corresponding upward motion of the 
# first P arrival at a site. The area of the T-axis on the focal sphere
# is usually colored (black).
# The P axis means compression at the source (and downward motion at sites).
# The P axis area is usually left white. 
# The T axis is associated with a positive eigenmoment and 
# the P axis with a negative eigenmoment. 
# When the T and P eigenmoments are equal (and of opposite sign) the
# focal mechanism is a shear dislocation and can be modeled with a 
# double couple moment tensor (or Moment, srike, dip, rake)
# Adding a CLVD component or a minor double couple adds a parameter.
# Adding an isotropic component adds another parameter. 
# The moment tensor is symmetric and thus has at most 6 parameters. 

# Aki & Richards coordinate system: 
# z (vertical, positive down)
# x (positive North)
# y (positive East)
# Normal mode coordinate system (used by Global CMT):
#  r (vertical, positive up)
#  theta (t), or delta (epicentral distance) (positive South)
#  phi (p or f) (positive East)
# Therefore,
# mt(1-6): Mrr, Mtt, Mff, Mrt, Mrf, Mtf
# mt(1-6): Mzz, Mxx, Myy, Mzx, -Mzy, -Mxy

######################################################

import numpy as np

eps = 1.e-6
halfpi = 0.5*np.pi 
twopi = 2*np.pi

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


def getplanes(xm):
    """
    IN: xm = list of moment tensor elements in normal-mode order (GCMT)
             i.e. Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
    OUT: trace, CLVD, m0 (without the scale)
         T, N, and P axes' azimuths and dips
         strike dip rake (twice, for both fault planes) 
    """
    xmatrix = np.array([[xm[0],xm[3],xm[4]],[xm[3],xm[1],xm[5]],[xm[4],xm[5],xm[2]]]) 
    tr = xm[0]+xm[1]+xm[2]  # trace of moment tensor
    third = tr/3.0
    if np.abs(tr)>eps:
       xmatrix[0,0] = xmatrix[0,0] - third
       xmatrix[1,1] = xmatrix[1,1] - third
       xmatrix[2,2] = xmatrix[2,2] - third
       #print('removed isotropic component from Moment Tensor:')
    d,ptn = np.linalg.eigh(xmatrix)  # eigenvectors are T, N, and P axes.
    jt = np.argmax(d) ; dmax = d[jt]  # find T axis 
    jp = np.argmin(d) ; dmin = d[jp]  # find P axis  
    for j in [0,1,2]:
        if j!=jp and j!=jt: jn=j    # find N axis
    if (jn+jp+jt)!=3:
        print('ERROR in axis determination')
        return 0
    p = ptn[:,jp]
    t = ptn[:,jt]
    n = ptn[:,jn]

    #print('t ',t)
    #print('n ',n)
    #print('p ',p)

    if -1.*d[jp]>d[jt]:   # find CLVD
       djpt = d[jp]
    else:
       djpt = d[jt]
    clvd = d[jn]/djpt
    m0 = 0.5*(np.abs(d[jp])+np.abs(d[jt]))  # set M0 (unscaled)
    #print('CLVD component = ', clvd)
    #print('Moment (unscaled) = ', m0)

    azt,dpt = azdp(t,'deg')
    azn,dpn = azdp(n,'deg')
    azp,dpp = azdp(p,'deg')
    (st1,dip1,rake1), (st2,dip2,rake2) = tp2sdr(t,p)

    st1 = np.degrees(st1)
    st2 = np.degrees(st2)
    dip1 = np.degrees(dip1)
    dip2 = np.degrees(dip2)
    rake1 = np.degrees(rake1)
    rake2 = np.degrees(rake2)

    #print( tr,clvd, m0)
    #print((azt,dpt),(azn,dpn),(azp,dpp))
    #print((st1,dip1,rake1), (st2,dip2,rake2))

    return tr,clvd, m0,\
           (azt,dpt),(azn,dpn),(azp,dpp),\
           (st1,dip1,rake1), (st2,dip2,rake2) 
           # angles are returned in degrees.


def getmt(moment,st,dp,rk):
    """
    IN: moment strike dip rake (assuming double couple mechanism) - in degrees
             each of these may be an array (given that we use numpy below)
    OUT: list of moment tensor elements in normal-mode order ((C)MT, follows 
                  Gilbert and Dziewonski (1975), Dziewonski et al. (1981), Dziewonski and Woodhouse, (1983)) 
             i.e. Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, exponent of the MT scale (10**exponent)
    units of MT elements are the same as the units for the input moment.
    formulas from Aki & Richards Box 4.4
    mt(1-6): Mrr, Mtt, Mff, Mrt, Mrf, Mtf
    mt(1-6): Mzz, Mxx, Myy, Mzx, -Mzy, -Mxy
    """
    st = np.radians(st)
    dp = np.radians(dp)
    rk = np.radians(rk)
    sd = np.sin(dp) 
    cd = np.cos(dp)
    sd2 = np.sin(2*dp) 
    cd2 = np.cos(2*dp)
    ss = np.sin(st) 
    cs = np.cos(st)
    ss2 = np.sin(2*st) 
    cs2 = np.cos(2*st)
    sr = np.sin(rk) 
    cr = np.cos(rk)

    logm = np.log10(moment)
    exponent = int(logm)   # scale of moment tensor (10**exponent)
    remain = logm-exponent
    m0 = 10**remain        # multiplier for MT elements

    Mrr = m0*(sr*sd2) 
    Mtt = m0*(-1.*sd*cr*ss2 - sd2*sr*ss*ss)
    Mff = m0*(sd*cr*ss2 - sd2*sr*cs*cs)
    Mrt = m0*(-1.*cd*cr*cs - cd2*sr*ss)
    Mrf = m0*(cd*cr*ss - cd2*sr*cs)
    Mtf = m0*(-1.*sd*cr*cs2 - 0.5*sd2*sr*ss2)

    return (Mrr, Mtt, Mff, Mrt, Mrf, Mtf), exponent


def RpatternDC(fault,azimuth,exit_angles):
    """
    Calculate predicted amplitudes of P, SV, and SH waves.
    IN: fault = [strike, dip, rake]
             = faulting mechanism, described by a list of strike, dip, and rake
             (note, strike is measure clockwise from N, dip is measured positive downwards
             (between 0 and 90) w.r.t. a horizontal that is 90 degrees clockwise from strike,
             and rake is measured positive upwards (counterclockwise)
        azimuth: azimuth with which ray path leaves source (clockwise from N)
        exit_angles = [i, j]
              i = angle between P ray path & vertical in the source model layer
              j = angle between S ray path & vertical in the source model layer
    OUT: Amplitudes for P, SV, and SH waves
    P as measured on L (~Z) component, SV measured on Q (~R) component, and SH measured on T component.
    All input is in degrees.
    Formulas are the far-field terms from Aki & Richards (1980).
    They have yet to be scaled with the reciprocals of hypocentral alpha and beta cubed.
    Common factors for all amplitudes, i.e. seismic moment and the 
    density (at hypocenter), geometrical spreading, layer-boundary, and attentuative properties 
    of the Martian interior remain unaccounted for.
    """

    strike,dip,rake = fault
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

    iP = np.radians(exit_angles[0])
    jS = np.radians(exit_angles[1])

    AP = sR*(3*np.cos(iP)**2 - 1) - qR*np.sin(2*iP) - pR*np.sin(iP)**2
    ASV = 1.5*sR*np.sin(2*jS) + qR*np.cos(2*jS) + 0.5*pR*np.sin(2*jS)
    ASH = -1*(-qL*np.cos(jS) - pL*np.sin(jS))  # inverted to align with observational seismology convention of SH
    # refer to Maddy's paper

    return AP,ASV,ASH  
    # these have yet to be scaled with the reciprocals of hypocentral alpha and beta cubed

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
#### define all the relevant functions #########

def F4R1(q1,q2,icode):
    """
    q=q2*(i,j,k,1))/q1 
    first compute q2*(i,j,k,1) by using boxtest
    second, get q usin quatD which divides two quaternions
    """
    ### compute q2*(i,j,k,1) bu calling boxtest
    qr2=my_BOXTEST(q2,icode)
    q=QUATD(q1,qr2)  #first element is the denominator and the second is the numerator
    qp=QUATP(q1,q)
    #print(f"q, {q}")
    rotation_angle,colatitude,azimuth=SPHCOOR(q)
    #print("anle,theta, phi")
    #print(rotation_angle, colatitude, azimuth)
    return q

def my_BOXTEST(q1,icode):
    """
    for Icode=0, finds the minimal rotation quaternion
    for Icode=N, N=1,3, finds the rotation quaternion q2=q1*(i,j,k,1)
    Note my code is for a quaternion q3*k+q2*j+q1*i+q0
    """
    q2 = np.zeros(4)
    quatt = np.zeros(4)
    quat=np.eye(4)
    q2=np.array([q1[i] for i in range(4)])
    if icode==0:
        #print("minimum rotation quaternion")
        quatt=quat[:,3]
    elif icode==1:
        quatt=quat[:,0]  #multiplication by i
    elif icode==2: 
        quatt=quat[:,1]  # multiplication by j
    elif icode==3: 
        quatt=quat[:,2]  #multiplication by k
    #print(quatt)
    q2=QUATP(quatt,q1)
    if q2[-1]<0:
        q2=np.array([-q2[i] for i in range(4)])
    #if icode==0:
     #   print(f"Qmin={q2}")
    return q2

def SPHCOOR(quat):
    """
    for the rotation quaternion quat, this function finds the rotation angle( angl) 
    of a counterclockwise rotation and spehrical coordinates (colatitude, theta, and azimuth)
    of the rotation pole (intersection of the axis with reference sphere); theta==0 corresponds
    to the vector pointing down.    
    """
    #print(quat)
    #quat=np.array(quat)
    q0=quat[-1]
    #print(q0)
    if q0<0.0:
        quat=[quat[i]*-1.0 for i in range(4)]
    quatn=np.sqrt(1.0-quat[-1]**2)
    costh=1.0
    if np.abs(quatn)>1.0e-10:
        costh=quat[2]/quatn
    if np.abs(costh)>1.0:
        costh=int(costh)
    colatitude=np.degrees(np.arccos(costh))
    #print(costh,theta)
    rotation_angle=np.degrees(2.0*np.arccos(quat[-1])) # rotation angle
    #print(quat[-1]) 
    azimuth=0.0
    if np.abs(quat[0])>1.0e-10 or np.abs(quat[1])>1.0e-10:
        azimuth=np.degrees(np.arctan2(quat[1],quat[0]))
        #print("azimuth is ")
        #print(azimuth)
    if azimuth <0.0:
        azimuth=azimuth+360.0
    return rotation_angle,colatitude,azimuth

def QUATP(quat1,quat2):
    """
    calculates the product of two quaternions q3=q2*q1,
    see F.klein v.1 p.61, or Altman, 1986, p.156,
    or Biedenham and Louck, 1981, p.185.
    quaternion is taken here as q1*i+q2*j+q3*k+q4
    """
    
    quat3=np.zeros(4)
    quat3[0]=quat1[3]*quat2[0]+quat1[2]*quat2[1]-quat1[1]*quat2[2]+quat1[0]*quat2[3]
    quat3[1]=-quat1[2]*quat2[0]+quat1[3]*quat2[1]+quat1[0]*quat2[2]+quat1[1]*quat2[3]
    quat3[2]=quat1[1]*quat2[0]-quat1[0]*quat2[1]+quat1[3]*quat2[2]+quat1[2]*quat2[3]
    quat3[3]=-quat1[0]*quat2[0]-quat1[1]*quat2[1]-quat1[2]*quat2[2]+quat1[3]*quat2[3]
    #quat3=[round(value, 4) for value in quat3]
    return quat3

def QUATD(q1,q2):
    """
    estimates the division of two quaternions quat1, and quat2,and saves it as quat3
    estimates (q2/q1)
    """
    #compute the conjugate of quat1
    qc1=q1.copy()   
    for i in range(3):
        qc1[i]=-q1[i]
    return QUATP(qc1,q2) 

def QUARTFPS(eqke,icode):
    """
    This function calculates rotation quaternion corresponding to earthquake focal mechanism
    icode==0: four input data: plunge and azimuth of T_axis
            plunge and azimuth of p-axis
    Since plunge and azimuth of 2 axes are redundant for calculation,
    (four degrees of freedom Vs three degrees that are necessary)
    and have low accuracy ( integer angular degrees), we calculate plane normal (V) and slip vector (s)
    axes, in order that all axes be orthogonal.
    icode==1: three input data: slip ange (sa), dip angle (da), dip direction (dd)
    perp variable checks the orthogonality of T- and P-axes, it should be small (0.01< or 50)
    """
    err=1.0e-15
    ic=1
    if icode==1:
        dd=np.deg2rad(eqke[0])
        da=np.deg2rad(eqke[1])
        sa=np.deg2rad(eqke[2])
        cdd=np.cos(dd)
        sdd=np.sin(dd)
        cda=np.cos(da)
        sda=np.sin(da)
        csa=np.cos(sa)
        ssa=np.sin(sa)
        s1=csa*sdd-ssa*cda*cdd
        s2=-csa*cdd-ssa*cda*sdd
        s3=-ssa*sda
        v1=sda*cdd
        v2=sda*sdd
        v3=-cda
    else:
        t_plunge,t_azim=np.deg2rad(eqke[0]),np.deg2rad(eqke[1])
        p_plunge,p_azim=np.deg2rad(eqke[2]),np.deg2rad(eqke[3])
        t1=np.cos(t_azim)*np.cos(t_plunge)
        t2=np.sin(t_azim)*np.cos(t_plunge)
        t3=np.sin(t_plunge)
        p1=np.cos(p_azim)*np.cos(p_plunge)
        p2=np.sin(p_azim)*np.cos(p_plunge)
        p3=np.sin(p_plunge)
        perp=t1*p1+t2*p2+t3*p3   # orthogonal to both t- and p, like the b-axis or null-axis
        #if perp>2.0e-2:
        #    print(eqke,t1,t2,t3,p1,p2,p3,perp)
        v1=t1+p1
        v2=t2+p2
        v3=t3+p3
        s1=t1-p1
        s2=t2-p2
        s3=t3-p3
        anormv=np.sqrt(v1**2+v2**2+v3**2)
        v1,v2,v3=v1/anormv,v2/anormv,v3/anormv
        anorms=np.sqrt(s1**2+s2**2+s3**2)
        s1,s2,s3=s1/anorms,s2/anorms,s3/anorms
    an1=s2*v3-v2*s3
    an2=v1*s3-s1*v3
    an3=s1*v2-v1*s2
    #sinv3=s1*v2*an3+s2*v3*an1+v1*an2*s3-s3*v2*an1-s1*an2*v3-an3*v1*s2
    t1=(v1+s1)/np.sqrt(2.0)
    t2=(v2+s2)/np.sqrt(2.0)
    t3=(v3+s3)/np.sqrt(2.0)
    p1=(v1-s1)/np.sqrt(2.0)
    p2=(v2-s2)/np.sqrt(2.0)
    p3=(v3-s3)/np.sqrt(2.0)
    #print(t1,t2,t3,p1,p2,p3,an1,an2,an3)
    u0=(t1+p2+an3+1.0)/4.00
    u1=(t1-p2-an3+1.00)/4.00
    u2=(-t1+p2-an3+1.00)/4.00
    u3=(-t1-p2+an3+1.00)/4.00
    vm=max(u0,u1,u2,u3)
    if vm==u0:
        icode=0*ic
        u0=np.sqrt(u0)
        u3=(t2-p1)/(4.00*u0)
        u2=(an1-t3)/(4.00*u0)
        u1=(p3-an2)/(4.00*u0)
    elif vm==u1:
        icode=1*ic
        u1=np.sqrt(u1)
        u2=(t2+p1)/(4.00*u1)
        u3=(an1+t3)/(4.00*u1)
        u0=(p3-an2)/(4.00*u1)
    elif vm==u2:
        icode=2*ic
        u2=np.sqrt(u2)
        u1=(t2+p1)/(4.00*u2)
        u0=(an1-t3)/(4.00*u2)
        u3=(p3+an2)/(4.00*u2)
    elif vm==u3:
        icode=3*ic
        u3=np.sqrt(u3)
        u0=(t2-p1)/(4.00*u3)
        u1=(an1+t3)/(4.00*u3)
        u2=(p3+an2)/(4.00*u3)
    temp=u0*u0+u1*u1+u2*u2+u3*u3
   # if np.abs(temp-1.0)>err:
    #    print("***error***")
    #    print(t1,t2,t3,p1,p2,p3)
    quat=[u1,u2,u3,u0]
    return quat
    
def kagan_angles(eqke1,eqke2,eqke_code):
    """
    Estimates the kagan rotation angles for the rotation of two focal mechanisms of two earthquakes
    Input: earthquake parameters eqke1 and eqke2. Each is a list with parameters
    [t-plunge,t-azimuth,P-plunge,P-azimuth] for which eqke_code!=1 or [slip-angle,dip,dip direction(azimuth)]
    for which eqke_code==1
    Output:
        kagan angles
        
    """
    angles1,angles2,angles3,angles4={},{},{},{}
    quat1=QUARTFPS(eqke1,eqke_code)
    quat2=QUARTFPS(eqke2,eqke_code)
    
    #print(quat1,quat2)
    #print()
    q10=F4R1(quat1,quat2,icode=0)
    rot_angle1,colat1,azi1=SPHCOOR(q10)
    angles1['q'],angles1['angles']=q10,[rot_angle1,colat1,azi1]
    #print(f"{q10} with icode=0") 
    #print() 
    q1A=F4R1(quat1,quat2,icode=1)
    rot_angle2,colat2,azi2=SPHCOOR(q1A)
    angles2['q'],angles2['angles']=q1A,[rot_angle2,colat2,azi2]
    #angles2=[rot_angle2,colat2,azi2]
    #print(f"{q1A} with icode=1") 
    q1B=F4R1(quat1,quat2,icode=2)
    #print(f"{q1B} with icode=2")
    rot_angle3,colat3,azi3=SPHCOOR(q1B)
    angles3['q'],angles3['angles']=q1B,[rot_angle3,colat3,azi3]
    #angles3=[rot_angle3,colat3,azi3]
    q1D=F4R1(quat1,quat2,icode=3)
    #print(f"{q1D} with icode=3")
    rot_angle4,colat4,azi4=SPHCOOR(q1D)
    angles4['q'],angles4['angles']=q1D,[rot_angle4,colat4,azi4]
    return angles1,angles2,angles3,angles4

def FPS4R(eqh1,eqh2):
    """
    """
    #run for the first earthquake eqh2
    quat1=QUARTFPS(eqh1,icode=0)
    #print(f"quaternion of rotation for {eqh1} is {quat1}")
    #print(quat1)
    
    rotation_angle,colatitude,azimuth=SPHCOOR(quat1)
    #print("rotation angle, colatitude, azimuth for eqke 1")
    #print(rotation_angle,colatitude,azimuth)
    quatr1=my_BOXTEST(quat1,icode=0)
    #print("quatr1")
    #print(quatr1)
    #print(f"qm {qm} and quatr1 {quatr1} icode=0")
    #print("                  ")
    rotation_angle,colatitude,azimuth=SPHCOOR(quatr1)
    #print("angl, theta,phi for eqke1")
    #print(f"{rotation_angle},{colatitude},{azimuth}")
    #run for the second earthquake eqh2
    quat2=QUARTFPS(eqh2,icode=0) 
    #print(f"quaternion of rotation for {eqh2}")
    #print(quat2)
    rotation_angle,colatitude,azimuth=SPHCOOR(quat2)
    #print("              ")
    #print("rotation angle, colatitude, azimuth for eqke2")
    #print(rotation_angle,colatitude,azimuth)
    quatr2=my_BOXTEST(quat2,icode=0)
    #print(f"qm {qm} and quatr2 {quatr2} icode=0")
    #print("                  ")
    rotation_angle,colatitude,azimuth=SPHCOOR(quatr2)
    #print("         ")
    #print("angl, theta,phi for eqke2")
    #print(rotation_angle,colatitude, azimuth)
    #print("      ")
    q10=F4R1(quat1,quat2,icode=0)
    #print(f"{q10} with icode=0")
    
    q1A=F4R1(quat1,quat2,icode=1)
    #print(f"{q1A} with icode=1")
    
    q1B=F4R1(quat1,quat2,icode=2)
    #print(f"{q1B} with icode=2")
    
    q1D=F4R1(quat1,quat2,icode=3)
    #print(f"{q1D} with icode=3")
    
    return quat1,quat2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371.0
    
    # Distance in kilometers
    distance = c * r
    return distance/111.19

#eqke1=[24,120,41,232] 
#eqke2=[55,295,17,51]
# call for earthquake 3 and earthquake 4 ###
#q1,q2=FPS4R(eqke1,eqke2)
def min_kagan_angle(eqke1, eqke2, source_type):
    """
    returns the minimum kagan rotation angle for any inputs eqke1 and eqke2. 
    source_type==1 if the earthquake parameters are strike, dip, and rake or any number!=1 if t-plunge, T-azimuth and P-plunge, P-azimuth. 
    eqke1=[strike1,dip1,rake1] for equake 1 
    eqke1=[strike1,dip1,rake1] for equake 2
    
    """
    angles1, angles2, angles3, angles4 = kagan_angles(eqke1, eqke2, source_type)
    rot_params = [angles1['angles'], angles2['angles'], angles3['angles'], angles4['angles']]
    # Find the list with the smallest first element
    min_params = min(rot_params, key=lambda x: x[0])
    
    # Find the set with the smallest first element
    min_set = None
    for angle_set in [angles1, angles2, angles3, angles4]:
        if angle_set['angles'] == min_params:
            min_set = angle_set
    return min_set
    
    
#min_params1=min_kagan_angle(eqke1, eqke2, source_type=3)
#print(min_params1)

## call the function to compute the 4 kagan angles
min_params2=min_kagan_angle([353,33,-78], [213,40,-74], source_type=1)
print(min_params2)

#######################################################################################################
file = pd.read_excel('kagan_event_params.xlsx', index_col=None)
azs=file.az
dips=file.dip
rakes=file.rake
lats=file.lat
lons=file.lon
time=file.time
year=file.year
####define a commom reference event  ref_event
ref_event=[33,8,-36] # just an example
angles=[]
distance=[]
for i,az in enumerate(azs):
	event2=[az,dips[i],rakes[i]]
	min_params=min_kagan_angle(ref_event, event2, source_type=1)
	#print()
	min_angle=round(min_params['angles'][0],2)
	angles.append(min_angle)
	delta=round(haversine_distance(-7.27, -71.49, lats[i], lons[i]),2)
	distance.append(delta)
	#print(year[i],time[i],lats[i],lons[i],delta,min_angle)
print(angles)
file["distance"]=distance
file["angles"]=angles
file.to_excel('kagan_out.xlsx', index=False)






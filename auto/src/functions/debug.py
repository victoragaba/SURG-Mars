'''
Functions for plotting beachballs.
This partially uses python code from ObsPy, which adapted code from bb.m
written by Andy Michael, Chen Ji and Oliver Boyd.

bb.m: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
Obspy: https://docs.obspy.org/packages/autogen/obspy.imaging.beachball.html
'''
# Standard libraries
import copy
import os

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def takeoff_az2xy(takeoff,azimuth,projection='lambert'):
    '''
    Projects takeoff and azimuths onto focal sphere.
    Supported projections are 'stereographic' and 'lambert'

    Takeoffs:
        >90: downgoing, <90: upgoing
    Azimuths:
        # 0: North, 90: east, etc.
    '''
    takeoff=180-takeoff
    r=np.ones(len(takeoff))
    r[takeoff>90]=-1
    
    theta=np.deg2rad(takeoff)
    phi=np.deg2rad(90-azimuth)
    
    xyz=np.empty((3,len(takeoff)),dtype=float)
    xyz[0,:]=r*np.sin(theta)*np.cos(phi)
    xyz[1,:]=r*np.sin(theta)*np.sin(phi)
    xyz[2,:]=r*np.cos(theta)
    
    if projection=='stereographic':
        xy=xyz[:2,:]/(1+xyz[2,:])
    elif projection=='lambert':
        xy=xyz[:2,:]/np.sqrt(1+xyz[2,:])
    else:
        raise ValueError('Unknown projection: {}'.format(projection))
    return xy.T


def takeoff_az2xy(takeoff, azimuth, projection='lambert'):
    '''
    Projects takeoff and azimuths onto focal sphere.
    Supported projections are 'stereographic' and 'lambert'

    Takeoffs:
        >90: downgoing, <90: upgoing
    Azimuths:
        # 0: North, 90: east, etc.
    '''
    takeoff = np.atleast_1d(takeoff)
    azimuth = np.atleast_1d(azimuth)

    theta = np.deg2rad(takeoff)
    phi = np.deg2rad(90 - azimuth)

    # Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    if projection == 'stereographic':
        denom = 1 + z
        denom[denom == 0] = np.nan
        xy = np.vstack((x / denom, y / denom))
    elif projection == 'lambert':
        denom = np.sqrt(1 + z)
        denom[denom == 0] = np.nan
        xy = np.vstack((np.sqrt(2) * x / denom,
                        np.sqrt(2) * y / denom))
    else:
        raise ValueError(f'Unknown projection: {projection}')

    return xy.T
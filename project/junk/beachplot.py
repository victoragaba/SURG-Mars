'''
Name: Victor Agaba

Date: 8th December 2024

The goal of this module is to provide all helper functions needed for
beachball plotting.
'''

import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach
import functions as fn


def grid_beach(solution_set, width, max_plot, facecolor):
    '''
    (ref Omkar)
    Plot complete solution set, with n figures in each plot
    Solution set is a list of [s, d, r] for each focal mechanism
    Input in DEGREES!!!
    '''
    # create figure
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    # hide axes and ticks
    ax.axison = False
     
    # plot all solutions
    for counter, solution in enumerate(solution_set):
        x = 210 * (counter % width)
        y = 210 * (counter // width)
        
        solution = np.rad2deg(fn.bound(solution))
        collection = beach(solution, xy=(x, y), facecolor=facecolor)
        ax.add_collection(collection)
        
        if counter == max_plot:
            break
    
    # scale and plot
    ax.autoscale_view()
    if facecolor == 'red': ax.set_title("Central mechanism solutions")
    else: ax.set_title("Best fitting solutions")
    
    plt.show()
    

def overlay_beach(solution_set, facecolor):
    '''
    (ref Suzan)
    Plot overlayed beachballs
    '''
    pass


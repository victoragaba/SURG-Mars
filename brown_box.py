from function_repo import *

"""
MY NOTES:

Trim this down to only the essentials, then use function-repo

My method is more computationally expensive

Reverse rigid sampling to fix the hole at the pole

Look into elliptical distributions
"""

if __name__ == '__main__':
    
    # pass

    ## GENERATING SAMPLES
    # N = 100
    # t_samples = hemisphere_samples(N**2)
    # sdr1 = []
    # sdr2 = []
    
    # for t in t_samples:
    #     p_rotations = uniform_samples(N, [0, np.pi])
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
    #     plt.ylabel(ylabels[i])
    # plt.xlabel("Angles (degrees)")
    
    # for i in range(3):
    #     plt.subplot(3,2,2*(i+1))
    #     plt.hist([np.rad2deg(emt[i]) for emt in sdr2], bins=50, density=True)
    #     if i == 0: plt.title("2nd fault plane")
    #     plt.ylabel(ylabels[i])
        
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
    old_epsilon = get_sphere_epsilon(Ao, Uo) # correct this
    
    # Grid search + my version of epsilon
    N = 100
    
    # old_accepted1 = [] # accepted fault planes within 1 standard deviation
    # old_accepted2 = [] # accepted fault planes within 2 standard deviations
    # old_rejected = []  # rejected fault planes
    # accepted1 = []; accepted2 = []; rejected = []
    
    filtered_sdrs = [[] for i in range(6)] # old and new accepted/rejected, same order
    # format per inner list: [([strike, dip, rake], weight), ...]
    
    # this takes a while
    t_samples = hemisphere_samples(N**2)
    for t in t_samples:
        p_rotations = uniform_samples(N, [0, np.pi])
        p_start = starting_direc(t, j_hat)
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
    
    # find a way to extract all strikes, dips and rakes into separate lists
    split_sdrs = dict()
    split_sdrs["Strikes"] = [[emt[0][0] for emt in filt] for filt in filtered_sdrs]
    split_sdrs["Dips"] = [[emt[0][1] for emt in filt] for filt in filtered_sdrs]
    split_sdrs["Rakes"] = [[emt[0][2] for emt in filt] for filt in filtered_sdrs]
    titles = ["Strikes", "Dips", "Rakes"]
    y_axes = ["Accepted (1 std)", "Accepted (2 std)", "Rejected (>2 std)"]
    # I might want to create a dataframe for this ^^
    # I still need bins - go to sdr tests
    
    for i in range(3): # strikes, dips and rakes
        # main figures
        fig = plt.figure(figsize=(15,12))
        plt.suptitle(titles[i])
        
        for j in range(3): # (old + new) accepted1, accepted2, rejected
            # old
            plt.subplot(3,2,2*j+1) # subplots are 1-indexed
            plt.hist(split_sdrs[titles[i]][j], bins=50)
            if j == 0: plt.title("Old")
            plt.ylabel(y_axes[j])
            if j == 2: plt.xlabel("Angles (degrees)")
            # new
            plt.subplot(3,2,2*(j+1))
            plt.hist(split_sdrs[titles[i]][j+3], bins=50)
            if j == 0: plt.title("New")
            plt.ylabel(y_axes[j])
            if j == 2: plt.xlabel("Angles (degrees)")
    
        plt.show()
    
    """
    Draw out the data structures for debugging
    """
    
    # Scatter (3D sdr space)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # fg = ax.scatter3D
    
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
    Plot beachballs for accepted fault planes ***
    What can beachballs show us that histograms can't? (Kagan angle)
    How is a best fitting solution chosen? (Maddy's paper)
    Look into regression methods -- are they necessary?
    How to visualize weights?
        scatter plot
        something with a color density
    How many families?
    t/p distributions?
    """
    
    """
    From group meeting:
    Visualization options:
    -3D scatter plot
    -Grid of 2D histograms
    -Stacked histogram?
    -Beachballs!
    -Color density/color coding
    """
    
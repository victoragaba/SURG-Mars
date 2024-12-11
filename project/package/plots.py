'''
Name: Victor Agaba

Date: 10th December 2024

The goal of this module is to implement plotting functions for
the data analysis.
'''

import numpy as np
from matplotlib import pyplot as plt
from obspy.imaging.beachball import beach
from . import functions as fn
from . import model as sm


def misfit(model: sm.SeismicModel):
    '''
    Plot the misfits in a subplot.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(model.get_misfits())
    ax.set_title('Misfit function')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misfit')
    plt.show()
    

def half_angles(model: sm.SeismicModel, bins=20):
    '''
    Plot the half angles in a histogram.
    This is a diagnostic plot.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(np.rad2deg(model.get_half_angles()), bins=bins)
    ax.set_title('Diagnostic histogram of half-angles')
    ax.set_xlabel('Half angle (deg)')
    ax.set_ylabel('Frequency')
    plt.show()
    
    
def iterates_2D(model: sm.SeismicModel, cmap='rainbow', s=10, optimal=True, index=2):
    '''
    Make 3 2D plots of the iterates: psi, delta, lambda.
    psi against delta, psi against lambda, delta against lambda.
    Join the points with a line and color them by iteration number.
    
    Args:
        optimal (bool): If True, plot only the optimal points.
        index (int): 0 is 1st fault plane, 1 is 2nd fault plane, 2 is both.
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    if index == 0 or index == 1:
        if not optimal:
            model.mirror(['optimals', 'iterates'], index)
            iterates = model.get_iterates()
            optimal_iterates = model.get_optimal_iterates()
        else:
            model.mirror(['optimals'], index)
            optimal_iterates = model.get_optimal_iterates()
    elif index == 2:
        iterates, optimal_iterates = [], []
        if not optimal:
            model.mirror(['optimals', 'iterates'], 0)
            iterates.extend(model.get_iterates())
            optimal_iterates.extend(model.get_optimal_iterates())
            model.mirror(['optimals', 'iterates'], 1)
            iterates.extend(model.get_iterates())
            optimal_iterates.extend(model.get_optimal_iterates())
        else:
            model.mirror(['optimals'], 0)
            optimal_iterates.extend(model.get_optimal_iterates())
            model.mirror(['optimals'], 1)
            optimal_iterates.extend(model.get_optimal_iterates())
    
    opt_strikes = [np.rad2deg(m[0]) for m in optimal_iterates]
    opt_dips = [np.rad2deg(m[1]) for m in optimal_iterates]
    opt_rakes = [np.rad2deg(m[2]) for m in optimal_iterates]
    
    # convert the angles to degrees
    if not optimal:
        strikes = [np.rad2deg(m[0]) for m in iterates]
        dips = [np.rad2deg(m[1]) for m in iterates]
        rakes = [np.rad2deg(m[2]) for m in iterates]
        weights = -np.array(model.misfits)
    else:
        weights = np.array(model.get_optimal_laplacians())
    if index == 2: weights = np.concatenate([weights, weights])
    
    # create a ScalarMappable for consistent colorbar scaling
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # make the plots
    if not optimal:
        ax1.scatter(strikes, dips, c=weights, cmap=cmap, norm=norm, s=s)
        ax1.scatter(opt_strikes, opt_dips, c='black', marker='*', s=s, label='Optimal')
    else:
        ax1.scatter(opt_strikes, opt_dips, c=weights, cmap=cmap, marker='*', s=s, label='Optimal')
    ax1.set_title('Strike against Dip')
    ax1.set_xlabel('Strike (deg)')
    ax1.set_ylabel('Dip (deg)')
    
    if not optimal:
        ax2.scatter(strikes, rakes, c=weights, cmap=cmap, norm=norm, s=s)
        ax2.scatter(opt_strikes, opt_rakes, c='black', marker='*', s=s, label='Optimal')
    else:
        ax2.scatter(opt_strikes, opt_rakes, c=weights, cmap=cmap, marker='*', s=s, label='Optimal')
    ax2.set_title('Strike against Rake')
    ax2.set_xlabel('Strike (deg)')
    ax2.set_ylabel('Rake (deg)')
    
    if not optimal:
        ax3.scatter(dips, rakes, c=weights, cmap=cmap, norm=norm, s=s)
        ax3.scatter(opt_dips, opt_rakes, c='black', marker='*', s=s, label='Optimal')
    else:
        ax3.scatter(opt_dips, opt_rakes, c=weights, cmap=cmap, marker='*', s=s, label='Optimal')
    ax3.set_title('Dip against Rake')
    ax3.set_xlabel('Dip (deg)')
    ax3.set_ylabel('Rake (deg)')
    
    # add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    if not optimal:
        cbar.set_label('Cosine similarity')
    else:
        cbar.set_label('Laplacian')
    
    plt.show()
    
    
def amplitudes(model: sm.SeismicModel, elev=30, azim=45, cmap='rainbow', s=10, alpha=0.5,
               iterates=False, observed=True):
    '''
    Make a 3D scatter plot of the optimal amplitudes.
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    optimal_amplitudes = model.get_optimal_amplitudes()
    opt_AP = [a[0] for a in optimal_amplitudes]
    opt_ASV = [a[1] for a in optimal_amplitudes]
    opt_ASH = [a[2] for a in optimal_amplitudes]
    
    if iterates:
        amplitudes = model.get_amplitudes()
        AP = [a[0] for a in amplitudes]
        ASV = [a[1] for a in amplitudes]
        ASH = [a[2] for a in amplitudes]
        weights = -np.array(model.misfits)
        
        # normalize weights for consistent coloring
        norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        cmap_instance = plt.cm.get_cmap(cmap)
        
        scatter = ax.scatter(
            AP, ASV, ASH,
            c=weights, cmap=cmap_instance, norm=norm, s=0.01*s, alpha=alpha
        )
    
    ax.scatter(
        opt_AP, opt_ASV, opt_ASH,
        c='black', marker='*', s=s, label='Optimal'
    )
    
    # add the origin
    ax.scatter(0, 0, 0, c='blue', marker='o', s=s, label='Origin')
    
    # add observed amplitudes
    if observed:
        ax.scatter(
            model.Ao[0], model.Ao[1], model.Ao[2],
            c='red', marker='o', s=s, label='Observed'
        )
    
    ax.set_xlabel('AP')
    ax.set_ylabel('ASV')
    ax.set_zlabel('ASH')
    ax.legend()
    plt.title('Optimal amplitudes', fontsize=15)
    
    if iterates:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('Cosine similarity')
    
    # adjust the view angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.show()    


def tp_axes(model: sm.SeismicModel, elev=30, azim=45, half=False, central=False):
    '''
    Make a 3D plot of the optimal tp axes.
    '''
    if central: to_plot = model.get_central_tps()
    else: to_plot = model.get_tp_axes()
    zero = np.zeros(3)
    
    # compute the central axis
    c, _ = fn.regression_axes(model.get_tp_axes())
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot first axis for label
    t, p = to_plot[0]
    if half: t_prime, p_prime = zero, zero
    else: t_prime, p_prime = -t, -p
    ax.plot([t[0], t_prime[0]], [t[1], t_prime[1]], [t[2], t_prime[2]],
                c='black', alpha=0.5, label="t axis")
    ax.plot([p[0], p_prime[0]], [p[1], p_prime[1]], [p[2], p_prime[2]],
            c='red', alpha=0.5, label="p axis")
    
    # plot the rest of the axes
    for i in range(1, len(to_plot)):
        t, p = to_plot[i]
        if half: t_prime, p_prime = zero, zero
        else: t_prime, p_prime = -t, -p
        ax.plot([t[0], t_prime[0]], [t[1], t_prime[1]], [t[2], t_prime[2]],
                c='black', alpha=0.5)
        ax.plot([p[0], p_prime[0]], [p[1], p_prime[1]], [p[2], p_prime[2]],
                c='red', alpha=0.5)
    
    if half: c_prime = zero
    else: c_prime = -c
    ax.plot([c[0], c_prime[0]], [c[1], c_prime[1]], [c[2], c_prime[2]],
            c='green', alpha=0.5, label="central", linewidth=3, linestyle='--')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title('Optimal tp axes', fontsize=15)
    
    # adjust the view angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.show()
    

def beachballs(model: sm.SeismicModel, central=False, order_by='strike',width=10,
               max_plot=50, facecolor='blue'):
    '''
    Plot beachballs for the optimal solutions.
    '''
    model.filter_outliers()
    og_set = model.optimal_iterates
    if central:
        solution_set = [fn.tp2sdr(t, p)[0] for t, p in model.get_central_tps()]
        facecolor = 'red'
    else: solution_set = model.get_optimal_iterates()
    
    assert len(solution_set) == len(og_set), f'Length mismatch'
    # get sorting order from og_set
    if order_by == 'strike': order = np.argsort([s[0] for s in og_set])
    elif order_by == 'dip': order = np.argsort([s[1] for s in og_set])
    elif order_by == 'rake': order = np.argsort([s[2] for s in og_set])
    
    # now sort solution_set
    solution_set = [solution_set[i] for i in order]
    
    # plot the beachballs
    grid_beach(solution_set, width, max_plot, facecolor)


######################################################################
# (MONTE-CARLO) PLOTS FOR UNCERTAINTY QUANTIFICATION
######################################################################


def sampled_amplitudes(model: sm.SeismicModel, cmap='rainbow', s=10, alpha=1, azimuth=45,
                       elevation=30):
    '''
    Make a 3D scatter plot of the sampled amplitudes.
    Weight them by cosine similarity with Ao.
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    sampled_amplitudes = model.get_sampled_amplitudes()
    AP = [As[0] for As in sampled_amplitudes]
    ASV = [As[1] for As in sampled_amplitudes]
    ASH = [As[2] for As in sampled_amplitudes]
    weights = np.array(model.get_sampled_weights())
    
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap_instance = plt.cm.get_cmap(cmap)
    
    scatter = ax.scatter(
        AP, ASV, ASH,
        c=weights, cmap=cmap_instance, norm=norm, s=s, alpha=alpha
    )
    
    # plot the max weight with a black star
    max_index = np.argmax(weights)
    ax.scatter(
        AP[max_index], ASV[max_index], ASH[max_index],
        c='black', marker='*', s=10*s, label='Max weight'
    )
    
    ax.set_xlabel('AP')
    ax.set_ylabel('ASV')
    ax.set_zlabel('ASH')
    plt.legend()
    plt.title('Sampled amplitudes', fontsize=15)
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Cosine similarity')
    
    # adjust the view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    plt.show()


def uncertainty_2D(model: sm.SeismicModel, cmap='rainbow', s=10, scale=0.5):
    '''
    Plot the uncertainty ellipsoid in the parameter space.
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    optimal_parameterizations = model.get_optimal_parameterizations()
    thetas = [np.rad2deg(p[0]) for p in optimal_parameterizations]
    phis = [np.rad2deg(p[1]) for p in optimal_parameterizations]
    half_angles = [np.rad2deg(p[2]) for p in optimal_parameterizations]
    weights = np.array(model.get_sampled_weights())
    max_index = np.argmax(weights)
    
    # create a ScalarMappable for consistent colorbar scaling
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    ax1.scatter(thetas, phis, c=weights, cmap=cmap, norm=norm, s=s)
    ax1.scatter(thetas[max_index], phis[max_index], c='black', marker='*', s=10*s,
                label='Max weight')
    ax1.set_title('Theta against Phi')
    
    # add laplacian flow to max index
    if scale != 0:
        theta_flow, phi_flow = scale*fn.unit_vec(model.get_laplacian_flow()[1:])
        ax1.plot([thetas[max_index], thetas[max_index] + np.rad2deg(theta_flow)],
                [phis[max_index], phis[max_index] + np.rad2deg(phi_flow)],
                c='black', linestyle='--', label='Laplacian flow')
    
    ax1.set_xlabel('Theta')
    ax1.set_ylabel('Phi')
    ax1.legend()
    
    ax2.scatter(thetas, half_angles, c=weights, cmap=cmap, norm=norm, s=s)
    ax2.scatter(thetas[max_index], half_angles[max_index], c='black', marker='*', s=10*s,
                label='Max weight')
    ax2.set_title('Theta against Half-angle')
    ax2.set_xlabel('Theta')
    ax2.set_ylabel('Half-angle')
    ax2.legend()
    
    ax3.scatter(phis, half_angles, c=weights, cmap=cmap, norm=norm, s=s)
    ax3.scatter(phis[max_index], half_angles[max_index], c='black', marker='*', s=10*s,
                label='Max weight')
    ax3.set_title('Phi against Half-angle')
    ax3.set_xlabel('Phi')
    ax3.set_ylabel('Half-angle')
    ax3.legend()
    
    # add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cosine similarity')
    
    plt.show()


def uncertainty_3D(model: sm.SeismicModel, elev=30, azim=45, cmap='rainbow', s=10, alpha=1):
    '''
    Plot the uncertainty ellipsoid in the parameter space.
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    thetas = [p[0] for p in model.optimal_parameterizations]
    phis = [p[1] for p in model.optimal_parameterizations]
    half_angles = [p[2] for p in model.optimal_parameterizations]
    weights = np.array(model.sampled_weights)
    max_index = np.argmax(weights)
    
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap_instance = plt.cm.get_cmap(cmap)
    
    scatter = ax.scatter(
        thetas, phis, half_angles,
        c=weights, cmap=cmap_instance, norm=norm, s=s, alpha=alpha
    )
    ax.scatter(
        thetas[max_index], phis[max_index], half_angles[max_index],
        c='black', marker='*', s=10*s, label='Max weight'
    )
    
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Half angle')
    ax.legend()
    plt.title('Optimal parameterizations', fontsize=15)
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Cosine similarity')
    
    plt.show()
    
    
def optimal_errors(model: sm.SeismicModel, bins=10):
    '''
    Plot the optimal errors in a histogram.
    This is a diagnostic plot.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(np.rad2deg(model.optimal_errors), bins=bins)
    ax.set_title('Diagnostic histogram of optimal errors')
    ax.set_xlabel('Error (deg)')
    ax.set_ylabel('Frequency')
    
    plt.show()


######################################################################
# BEACHBALL PLOT HELPERS
######################################################################


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


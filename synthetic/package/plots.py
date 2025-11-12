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
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import art3d


def misfit(model: sm.RadiationModel):
    '''
    Plot the misfits in a subplot.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(model.get_misfits())
    ax.set_title('Misfit function')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misfit')
    plt.show()
    

def half_angles(model: sm.RadiationModel, bins=20):
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
    

def cov_ellipse(ax, x, y, cov_xy, conf=0.95, **kwargs):
    """
    Plot a covariance ellipse on a given Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the ellipse on.
    x, y : float
        Mean coordinates (center of the ellipse).
    cov_xy : (2, 2) array-like
        Covariance matrix for (x, y).
    conf : float, optional (default=0.95)
        Confidence level for the ellipse (e.g., 0.68, 0.95, 0.99).
    **kwargs :
        Additional keyword arguments passed to matplotlib.patches.Ellipse,
        e.g. edgecolor, facecolor, lw, alpha, linestyle.

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The created ellipse patch (already added to the axes).
    """
    # cov_xy = np.asarray(cov_xy)
    if cov_xy.shape != (2, 2):
        raise ValueError("cov_xy must be a 2x2 covariance matrix")

    # confidence scaling factor from chi-squared distribution
    scale = np.sqrt(chi2.ppf(conf, df=2))

    # eigen-decomposition to find ellipse parameters
    vals, vecs = np.linalg.eigh(cov_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # compute ellipse radii (2 * sqrt of eigenvalues gives diameters)
    width, height = 2 * scale * np.sqrt(vals)

    # angle of rotation in degrees
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # create and add ellipse
    ell = Ellipse( xy=(x, y), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)


def iterates_2D(model: sm.RadiationModel, cmap='rainbow', s=10, optimal=True,
                index=2, uncertainty=False, conf=0.95):
    '''
    Make 3 2D plots of the iterates: psi, delta, lambda.
    psi against delta, psi against lambda, delta against lambda.
    Join the points with a line and color them by iteration number.
    
    Args:
        optimal (bool): If True, plot only the optimal points.
        index (int): 0 is 1st fault plane, 1 is 2nd fault plane, 2 is both.
        uncertainty (bool): If True, include uncertainty ellipses. Only when optimal is True.
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # store observed sdr pair
    Ao_t, Ao_p = model.get_Ao_tp()
    if Ao_t is not None:
        Ao_sdr1, Ao_sdr2 = fn.tp2sdr(Ao_t, Ao_p, deg=True)
        Ao_sdrs = [Ao_sdr1, Ao_sdr2]
            
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
    
    # include uncertainty if specified
    if uncertainty:
        covariances = model.get_optimal_covariances()
        strike_dip_cov = []
        strike_rake_cov = []
        dip_rake_cov = []
        for cov in covariances:
            sd, sr, dr = fn.covariance_transform(cov)
            strike_dip_cov.append(sd)
            strike_rake_cov.append(sr)
            dip_rake_cov.append(dr)
    
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
        weights = np.ones(len(opt_strikes))  # no colorbar for optimal only
    
    # create a ScalarMappable for consistent colorbar scaling
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # make the plots
    if not optimal:
        ax1.scatter(strikes, dips, c=weights, cmap=cmap, norm=norm, s=s)
        ax1.scatter(opt_strikes, opt_dips, c='black', marker='.', s=s, label='Optimal')
    else:
        ax1.scatter(opt_strikes, opt_dips, c=weights, cmap=cmap, marker='.', s=s, label='Optimal')
        if Ao_t is not None:
            if index == 0 or index == 1:
                ax1.scatter(Ao_sdrs[index][0], Ao_sdrs[index][1], c='red', marker='o', s=s,
                            label='Observed')
            elif index == 2:
                ax1.scatter(Ao_sdrs[0][0], Ao_sdrs[0][1], c='red', marker='o', s=s,
                            label='Observed')
                ax1.scatter(Ao_sdrs[1][0], Ao_sdrs[1][1], c='red', marker='o', s=s)
        if uncertainty:
            for i in range(len(opt_strikes)):
                i_mirror = i % (len(opt_strikes)//2)
                cov_ellipse(
                    ax1, opt_strikes[i], opt_dips[i], strike_dip_cov[i_mirror], conf=conf,
                    edgecolor='black', alpha=0.3
                )
                
    ax1.set_title('Strike against Dip')
    ax1.set_xlabel('Strike (deg)')
    ax1.set_ylabel('Dip (deg)')
    
    if uncertainty:
        plt.suptitle(f'Optimal fault parameters with {conf*100:.0f}% confidence ellipses', fontsize=16)
    else:
        plt.suptitle('Optimal fault parameters', fontsize=16)
    
    if not optimal:
        ax2.scatter(strikes, rakes, c=weights, cmap=cmap, norm=norm, s=s)
        ax2.scatter(opt_strikes, opt_rakes, c='black', marker='.', s=s, label='Optimal')
    else:
        ax2.scatter(opt_strikes, opt_rakes, c=weights, cmap=cmap, marker='.', s=s, label='Optimal')
        if Ao_t is not None:
            if index == 0 or index == 1:
                ax2.scatter(Ao_sdrs[index][0], Ao_sdrs[index][2], c='red', marker='o', s=s,
                            label='Observed')
            elif index == 2:
                ax2.scatter(Ao_sdrs[0][0], Ao_sdrs[0][2], c='red', marker='o', s=s,
                            label='Observed')
                ax2.scatter(Ao_sdrs[1][0], Ao_sdrs[1][2], c='red', marker='o', s=s)
        if uncertainty:
            for i in range(len(opt_strikes)):
                i_mirror = i % (len(opt_strikes)//2)
                cov_ellipse(
                    ax2, opt_strikes[i], opt_rakes[i],
                    strike_rake_cov[i_mirror], edgecolor='black', alpha=0.3
                )
                
    ax2.set_title('Strike against Rake')
    ax2.set_xlabel('Strike (deg)')
    ax2.set_ylabel('Rake (deg)')
    
    if not optimal:
        ax3.scatter(dips, rakes, c=weights, cmap=cmap, norm=norm, s=s)
        ax3.scatter(opt_dips, opt_rakes, c='black', marker='.', s=s, label='Optimal')
    else:
        ax3.scatter(opt_dips, opt_rakes, c=weights, cmap=cmap, marker='.', s=s, label='Optimal')
        if Ao_t is not None:
            if index == 0 or index == 1:
                ax3.scatter(Ao_sdrs[index][1], Ao_sdrs[index][2], c='red', marker='o', s=s,
                            label='Observed')
            elif index == 2:
                ax3.scatter(Ao_sdrs[0][1], Ao_sdrs[0][2], c='red', marker='o', s=s,
                            label='Observed')
                ax3.scatter(Ao_sdrs[1][1], Ao_sdrs[1][2], c='red', marker='o', s=s)
        if uncertainty:
            for i in range(len(opt_dips)):
                i_mirror = i % (len(opt_dips)//2)
                cov_ellipse(
                    ax3, opt_dips[i], opt_rakes[i],
                    dip_rake_cov[i_mirror], edgecolor='black', alpha=0.3
                )
                
    ax3.set_title('Dip against Rake')
    ax3.set_xlabel('Dip (deg)')
    ax3.set_ylabel('Rake (deg)')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    if not optimal:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Cosine similarity')
        
    # set fixed axis limits for better comparison
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 90)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-180, 180)
    ax3.set_xlim(0, 90)
    ax3.set_ylim(-180, 180)

    plt.show()


def cov_ellipsoid(ax, mean, cov_matrix, conf, num_pts=50, **kwargs):
    '''
    Plot a 3D covariance ellipsoid on a given Matplotlib Axes3D.
    mean: (3,) array-like, center of the ellipsoid.
    cov_matrix: (3,3) array-like, covariance matrix.
    conf: confidence level for the ellipsoid (e.g., 0.68, 0.95, 0.99).
    **kwargs: additional keyword arguments passed to plot_wireframe.
    '''

    # confidence scaling factor from chi-squared distribution
    scale = np.sqrt(chi2.ppf(conf, df=3))

    # eigen-decomposition to find ellipsoid parameters
    vals, vecs = np.linalg.eigh(cov_matrix)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # create a grid of points on a unit sphere
    u = np.linspace(0, 2 * np.pi, num_pts)
    v = np.linspace(0, np.pi, num_pts)
    x = scale * np.sqrt(vals[0]) * np.outer(np.cos(u), np.sin(v))
    y = scale * np.sqrt(vals[1]) * np.outer(np.sin(u), np.sin(v))
    z = scale * np.sqrt(vals[2]) * np.outer(np.ones_like(u), np.cos(v))

    # rotate the points to align with the covariance axes
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot(vecs, [x[i,j], y[i,j], z[i,j]]) + mean
            
    # plot the ellipsoid
    ax.plot_wireframe(x, y, z, **kwargs)


def amplitudes(model: sm.RadiationModel, elev=30, azim=45, cmap='rainbow', s=50, alpha=0.5,
               iterates=False, observed=True, grid=False, cross_section: list = [0,0,0],
               eps_factor=1e-2, optimal=True, uncertainty=False, conf=0.95):
    '''
    Make a 3D scatter plot of the optimal amplitudes.
    cross_section: number of cross sections per axis (x, y, z).
    '''
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if optimal:
        optimal_amplitudes = model.get_optimal_amplitudes()
        opt_AP = [a[0] for a in optimal_amplitudes]
        opt_ASV = [a[1] for a in optimal_amplitudes]
        opt_ASH = [a[2] for a in optimal_amplitudes]
    
    if grid:
        grid_amplitudes = model.get_grid_amplitudes()
        grid_AP = [a[0] for a in grid_amplitudes]
        grid_ASV = [a[1] for a in grid_amplitudes]
        grid_ASH = [a[2] for a in grid_amplitudes]

        # plot the scatter graph
        ax.scatter(
            grid_AP, grid_ASV, grid_ASH,
            c='grey', marker='o', s=0.01*s, alpha=0.5*alpha, label='Grid points'
        )
    
    if iterates:
        amplitudes = model.get_amplitudes()
        AP = [a[0] for a in amplitudes]
        ASV = [a[1] for a in amplitudes]
        ASH = [a[2] for a in amplitudes]
        weights = -np.array(model.misfits)
        
        # normalize weights for consistent coloring
        norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        cmap_instance = plt.cm.get_cmap(cmap)
        
        it_scatter = ax.scatter(
            AP, ASV, ASH,
            c=weights, cmap=cmap_instance, norm=norm, s=0.01*s, alpha=alpha
        )
    
    if optimal:
        ax.scatter(
            opt_AP, opt_ASV, opt_ASH,
            c='black', marker='*', s=s, label='Optimal'
        )
    
    # add the origin
    ax.scatter(0, 0, 0, c='blue', marker='o', s=0.1*s, label='Origin')
    
    # add observed amplitudes
    if observed:
        ax.scatter(
            model.Ao[0], model.Ao[1], model.Ao[2],
            c='red', marker='o', s=0.5*s, label='Observed'
        )
    
        if uncertainty:
            Uo = model.get_Uo()
            # make covariance matrix
            cov_matrix = np.diag(Uo**2)
            
            # plot the uncertainty ellipsoid
            cov_ellipsoid(
                ax, model.get_Ao(), cov_matrix, conf=conf,
                edgecolor='red', alpha=0.1, label=f'{conf*100:.0f}% CR'
            )
    
    ax.set_xlabel('AP')
    ax.set_ylabel('ASV')
    ax.set_zlabel('ASH')
    ax.legend()
    if uncertainty:
        plt.title(f'Optimal amplitudes with {conf*100:.0f}% confidence ellipsoid', fontsize=15)
    else:
        plt.title('Optimal amplitudes', fontsize=15)
    
    if iterates:
        cbar = fig.colorbar(it_scatter, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('Cosine similarity')
    
    # adjust the view angle
    ax.view_init(elev=elev, azim=azim)
    
    # add cross sections, equally dividing the range of values
    # start with max and min vals available per axis
    if cross_section != [0,0,0]:
        all_AP, all_ASV, all_ASH = [], [], []
        if optimal:
            all_AP.extend(opt_AP)
            all_ASV.extend(opt_ASV)
            all_ASH.extend(opt_ASH)
        if iterates:
            all_AP.extend(AP)
            all_ASV.extend(ASV)
            all_ASH.extend(ASH)
        if observed:
            all_AP.append(model.Ao[0])
            all_ASV.append(model.Ao[1])
            all_ASH.append(model.Ao[2])
        if grid:
            all_AP.extend(grid_AP)
            all_ASV.extend(grid_ASV)
            all_ASH.extend(grid_ASH)

        min_AP, max_AP = min(all_AP), max(all_AP)
        min_ASV, max_ASV = min(all_ASV), max(all_ASV)
        min_ASH, max_ASH = min(all_ASH), max(all_ASH)
    
    plt.show()
    
    # make 2D scatter plots per cross section axis based on above
    if cross_section != [0,0,0]:
        if cross_section[0] > 0:
            num_x_sections = cross_section[0]
            eps_window = eps_factor*(max_AP - min_AP)
            x_sections = np.linspace(min_AP, max_AP, num_x_sections+2)[1:-1]
            
            fig, axes = plt.subplots(1, num_x_sections, figsize=(6*num_x_sections, 5))
            for idx, x in enumerate(x_sections):
                if optimal:
                    x_opt_points = [i for i, val in enumerate(opt_AP) 
                                    if x-eps_window <= val <= x+eps_window]
                    axes[idx].scatter(
                        [opt_ASV[i] for i in x_opt_points],
                        [opt_ASH[i] for i in x_opt_points],
                        c='black', marker='*', s=s, label='Optimal'
                    )
                if grid:
                    x_grid_points = [i for i, val in enumerate(grid_AP) 
                                     if x-eps_window <= val <= x+eps_window]
                    axes[idx].scatter(
                        [grid_ASV[i] for i in x_grid_points],
                        [grid_ASH[i] for i in x_grid_points],
                        c='grey', marker='o', s=0.1*s, alpha=0.5*alpha, label='Grid points'
                    )
                if iterates:
                    x_iterate_points = [i for i, val in enumerate(AP) 
                                        if x-eps_window <= val <= x+eps_window]
                    axes[idx].scatter(
                        [AP[i] for i in x_iterate_points],
                        [ASH[i] for i in x_iterate_points],
                        c=weights[[i for i in x_iterate_points]],
                        cmap=cmap, norm=norm, marker='o', s=0.1*s, alpha=alpha, label='Iterates'
                    )
                axes[idx].set_title(f'Cross section at AP = {x:.4f}')
                axes[idx].set_xlabel('ASV')
                axes[idx].set_ylabel('ASH')
                axes[idx].legend()
                
            plt.suptitle('Cross sections fixing AP', fontsize=16)
            plt.subplots_adjust(top=0.85)
            plt.show()
            
        if cross_section[1] > 0:
            num_y_sections = cross_section[1]
            eps_window = eps_factor*(max_ASV - min_ASV)
            y_sections = np.linspace(min_ASV, max_ASV, num_y_sections+2)[1:-1]

            fig, axes = plt.subplots(1, num_y_sections, figsize=(6*num_y_sections, 5))
            for idx, y in enumerate(y_sections):
                if optimal:
                    y_opt_points = [i for i, val in enumerate(opt_ASV) 
                                if y-eps_window <= val <= y+eps_window]
                    axes[idx].scatter(
                        [opt_AP[i] for i in y_opt_points],
                        [opt_ASH[i] for i in y_opt_points],
                        c='black', marker='*', s=s, label='Optimal'
                    )
                if grid:
                    y_grid_points = [i for i, val in enumerate(grid_ASV)
                                if y-eps_window <= val <= y+eps_window]
                    axes[idx].scatter(
                        [grid_AP[i] for i in y_grid_points],
                        [grid_ASH[i] for i in y_grid_points],
                        c='grey', marker='o', s=0.1*s, alpha=0.5*alpha, label='Grid points'
                    )
                if iterates:
                    y_iterate_points = [i for i, val in enumerate(ASV)
                                        if y-eps_window <= val <= y+eps_window]
                    axes[idx].scatter(
                        [AP[i] for i in y_iterate_points if i < len(AP)],
                        [ASH[i] for i in y_iterate_points if i < len(AP)],
                        c=weights[[i for i in y_iterate_points if i < len(AP)]],
                        cmap=cmap, norm=norm, marker='o', s=0.1*s, alpha=alpha, label='Iterates'
                    )
                axes[idx].set_title(f'Cross section at ASV = {y:.4f}')
                axes[idx].set_xlabel('AP')
                axes[idx].set_ylabel('ASH')
                axes[idx].legend()
            
            plt.suptitle('Cross sections fixing ASV', fontsize=16)
            plt.subplots_adjust(top=0.85)
            plt.show()

        if cross_section[2] > 0:
            num_z_sections = cross_section[2]
            eps_window = eps_factor*(max_ASH - min_ASH)
            z_sections = np.linspace(min_ASH, max_ASH, num_z_sections+2)[1:-1]

            fig, axes = plt.subplots(1, num_z_sections, figsize=(6*num_z_sections, 5))
            for idx, z in enumerate(z_sections):
                if optimal:
                    z_opt_points = [i for i, val in enumerate(opt_ASH) 
                                if z-eps_window <= val <= z+eps_window]
                    axes[idx].scatter(
                        [opt_AP[i] for i in z_opt_points],
                        [opt_ASV[i] for i in z_opt_points],
                        c='black', marker='*', s=s, label='Optimal'
                    )
                if grid:
                    z_grid_points = [i for i, val in enumerate(grid_ASH)
                                if z-eps_window <= val <= z+eps_window]
                    axes[idx].scatter(
                        [grid_AP[i] for i in z_grid_points],
                        [grid_ASV[i] for i in z_grid_points],
                        c='grey', marker='o', s=0.1*s, alpha=0.5*alpha, label='Grid points'
                    )
                if iterates:
                    z_iterate_points = [i for i, val in enumerate(ASH)
                                        if z-eps_window <= val <= z+eps_window]
                    axes[idx].scatter(
                        [AP[i] for i in z_iterate_points if i < len(AP)],
                        [ASV[i] for i in z_iterate_points if i < len(AP)],
                        c=weights[[i for i in z_iterate_points if i < len(AP)]],
                        cmap=cmap, norm=norm, marker='o', s=0.1*s, alpha=alpha, label='Iterates'
                    )
                axes[idx].set_title(f'Cross section at ASH = {z:.4f}')
                axes[idx].set_xlabel('AP')
                axes[idx].set_ylabel('ASV')
                axes[idx].legend()

            plt.suptitle('Cross sections fixing ASH', fontsize=16)
            plt.subplots_adjust(top=0.85)
            plt.show()


def tp_axes(model: sm.RadiationModel, elev=30, azim=45, half=False):
    '''
    Make a 3D plot of the optimal tp axes.
    
    Args:
        half (bool): If True, plot only half the axes from origin to tp.
    '''
    if len(model.get_tp_axes()) == 0: model.mirror(['axes'])
    
    to_plot = model.get_tp_axes()
    zero = np.zeros(3)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot first axis for label
    t, p = to_plot[0]
    if half: t_prime, p_prime = zero, zero
    else: t_prime, p_prime = -t, -p
    ax.plot([t[0], t_prime[0]], [t[1], t_prime[1]], [t[2], t_prime[2]],
                c='blue', alpha=0.5, label="t axis")
    ax.plot([p[0], p_prime[0]], [p[1], p_prime[1]], [p[2], p_prime[2]],
            c='black', alpha=0.5, label="p axis")
    
    # plot the rest of the axes
    for i in range(1, len(to_plot)):
        t, p = to_plot[i]
        if half: t_prime, p_prime = zero, zero
        else: t_prime, p_prime = -t, -p
        ax.plot([t[0], t_prime[0]], [t[1], t_prime[1]], [t[2], t_prime[2]],
                c='blue', alpha=0.5)
        ax.plot([p[0], p_prime[0]], [p[1], p_prime[1]], [p[2], p_prime[2]],
                c='black', alpha=0.5)
    
    # plot observed axes
    Ao_t, Ao_p = model.get_Ao_tp()
    if Ao_t is not None:        
        if half: Ao_t_prime, Ao_p_prime = zero, zero
        else: Ao_t_prime, Ao_p_prime = -Ao_t, -Ao_p
        ax.plot([Ao_t[0], Ao_t_prime[0]], [Ao_t[1], Ao_t_prime[1]], [Ao_t[2], Ao_t_prime[2]],
                c='red', alpha=1, label="Original")
        ax.plot([Ao_p[0], Ao_p_prime[0]], [Ao_p[1], Ao_p_prime[1]], [Ao_p[2], Ao_p_prime[2]],
                c='red', alpha=1)
    
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Optimal tp axes', fontsize=15)
    
    # adjust the view angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.show()
    

def beachballs(model: sm.RadiationModel, order_by='strike',width=10, max_plot=50,
               facecolor='blue', figsize=(15,10), overlap=False, original=False, uncertainty=False):
    '''
    Plot beachballs for the optimal solutions.
    order_by: 'strike', 'dip' or 'rake' to order the beachballs.
    width: number of beachballs per row.
    max_plot: maximum number of beachballs to plot.
    facecolor: color of the beachballs.
    '''
    if original:  # only when synthetic data is used
        Ao_t, Ao_p = model.get_Ao_tp()
        if order_by == 'tensor':
            Ao_mt = fn.tp2mt(Ao_t, Ao_p)
            Ao_comp_mt = np.array([
                Ao_mt[0,0], Ao_mt[1,1], Ao_mt[2,2],
                Ao_mt[0,1], Ao_mt[0,2], Ao_mt[1,2],
            ])
        else: Ao_sdr = fn.tp2sdr(Ao_t, Ao_p, deg=False)[0]
    
    assert order_by in ['strike', 'dip', 'rake', 'tensor'], f'Invalid order_by value'
    if order_by == 'tensor':
        og_set = model.get_hidden_optimal_iterates()
        solution_set = []
        eigen_set = []
        az = model.get_azimuth()
        for params in og_set:
            tensor = fn.hidden2mt(params, az, in_deg=False)
            tensor = fn.enu2use(tensor)
            compressed_tensor = np.array([
                tensor[0,0], tensor[1,1], tensor[2,2],
                tensor[0,1], tensor[0,2], tensor[1,2],
            ])
            solution_set.append(compressed_tensor)
            evals = fn.get_evals(tensor)
            eigen_set.append(np.abs(evals[1]))  # middle eigenvalue absolute
        # now sort solution_set in ascending order of middle eigenvalue
        order = np.argsort(eigen_set)
        solution_set = [solution_set[i] for i in order]
        if original: solution_set = [Ao_comp_mt] + solution_set
    else:
        og_set = model.get_optimal_iterates()
        solution_set = model.get_optimal_iterates() # optimal_covariances
        
        assert len(solution_set) == len(og_set), f'Length mismatch'
        # get sorting order from og_set
        if order_by == 'strike': order = np.argsort([s[0] for s in og_set])
        elif order_by == 'dip': order = np.argsort([s[1] for s in og_set])
        elif order_by == 'rake': order = np.argsort([s[2] for s in og_set])
        
        # now sort solution_set
        solution_set = [solution_set[i] for i in order]
        if original: solution_set = [Ao_sdr] + solution_set
    
    # plot the beachballs
    if overlap:
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=figsize)
        ax.axison = False
        solution_set = [np.rad2deg(s) for s in solution_set] if order_by != 'tensor' else solution_set
        collections = overlap_beach(solution_set)
        for collection in collections:
            ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_title("Overlapping beachball solutions")
        plt.show()
    else:
        covariances = []
        if uncertainty:
            assert order_by != 'tensor', "Uncertainty not supported for tensor beachballs"
            covariances = model.get_optimal_covariances() # NOTE
            # sort covariances in the same order as solution_set
            # BUG here
            covariances = [covariances[i] for i in order]
            # add original covariance:
            if original:
                covariances = [model.get_Ao_sdr_cov()] + covariances
        grid_beach(solution_set, width, max_plot, facecolor, figsize=figsize,
                   original=original, covariances=covariances)





######################################################################
# BEACHBALL PLOT HELPERS
######################################################################


def grid_beach(solution_set, width, max_plot, facecolor, figsize, original, covariances=[]):
    '''
    (ref Omkar)
    Plot complete solution set, with n figures in each plot
    Solution set is a list of [s, d, r] for each focal mechanism
    Input in DEGREES!!!
    '''
    # create figure
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=figsize)
    
    # hide axes and ticks
    ax.axison = False
    
    # check if I'm using tensor
    tensor = False
    if len(solution_set) > 0 and len(solution_set[0]) == 6: tensor = True
    
    if len(covariances) > 0: colors = fn.condition_colors(covariances)
    
    # plot all solutions
    for counter, solution in enumerate(solution_set):
        # x and y are coordinates of the beachball to align
        x = 210 * (counter % width)
        y = 210 * (counter // width)
        
        # if not tensor:
        #     solution = np.rad2deg(fn.bound(solution))
            
        if original and counter == 0:
            if len(covariances) == 0:
                if not tensor: solution = np.rad2deg(fn.bound(solution))
                collection = beach(solution, xy=(x, y), facecolor='red')
                ax.add_collection(collection)
            else:
                # sample 20 beachballs from a multivariate normal distribution
                samples = np.random.multivariate_normal(mean=solution,
                                                        cov=covariances[counter],
                                                        size=25)
                samples = [np.rad2deg(fn.bound(s)) for s in samples]
                collections = overlap_beach(samples, facecolor='red', xy=(x, y),
                                            alpha=0.1)
                for collection in collections:
                    ax.add_collection(collection)
        else:
            if len(covariances) == 0:
                if not tensor: solution = np.rad2deg(fn.bound(solution))
                collection = beach(solution, xy=(x, y), facecolor=facecolor)
                ax.add_collection(collection)
            else:
                # sample 20 beachballs from a multivariate normal distribution
                samples = np.random.multivariate_normal(mean=solution,
                                                        cov=covariances[counter],
                                                        size=25)
                samples = [np.rad2deg(fn.bound(s)) for s in samples]
                collections = overlap_beach(samples, facecolor=colors[counter], xy=(x, y),
                                            alpha=0.1)
                for collection in collections:
                    ax.add_collection(collection)
        
        if counter == max_plot: break
    
    # scale and plot
    ax.autoscale_view()
    if len(covariances) > 0: ax.set_title("Best fitting beachball solutions with uncertainty")
    else: ax.set_title("Best fitting beachball solutions")
    
    plt.show()


def overlap_beach(solution_set, facecolor='b', xy=(0, 0), alpha=0.05, max_plot=None):
    """
    Plot multiple overlapping beachballs at the same location.
    
    Parameters
    ----------
    solution_set : list
        List of focal mechanisms; each element can be [strike, dip, rake] or
        [M11, M22, M33, M12, M13, M23].
    xy : tuple
        Coordinates (x, y) of the beachball center.
    facecolor : str
        Color for the beachballs (default 'b').
    alpha : float
        Transparency for overlap visualization (default 0.5).
    figsize : tuple
        Figure size in inches (default (4, 4)).
    max_plot : int or None
        Maximum number of solutions to plot.
    """
    collections = []
    
    # plot all solutions overlapping
    for counter, solution in enumerate(solution_set):
        if max_plot is not None and counter >= max_plot:
            break
        
        # draw beachball at the same (x, y)
        collection = beach(solution, xy=xy, facecolor=facecolor, bgcolor='w',
                           edgecolor=None, alpha=alpha, nofill=False,
                           zorder=100, axes=None)
        collections.append(collection)
        
    return collections









































######################################################################
# (MONTE-CARLO) PLOTS FOR UNCERTAINTY QUANTIFICATION
######################################################################


def sampled_amplitudes(model: sm.RadiationModel, cmap='rainbow', s=10, alpha=1, azimuth=45,
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


def uncertainty_2D(model: sm.RadiationModel, cmap='rainbow', s=10, scale=0.5):
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


def uncertainty_3D(model: sm.RadiationModel, elev=30, azim=45, cmap='rainbow', s=10, alpha=1):
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
    
    
def optimal_errors(model: sm.RadiationModel, bins=10):
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


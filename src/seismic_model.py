'''
Name: Victor Agaba

Date: 2nd November 2024

The goal of this module is to create a model of the nonlinear inverse
problem to which we will apply the optimization algorithms.
'''


import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as fn
import optimizer as opt
import beachplot as bp


class Model:
    '''
    Base class for an unconstrained optimzation problem.
    Catch anything whose method is not implemented.
    '''
    
    def __call__(self):
        raise Exception('Class is not callable')

    def gradient(self):
        raise Exception('Method "gradient" is not implemented')

    def hessian(self):
        raise Exception('Method "hessian" not is implemented')

    def starting_point(self):
        raise Exception('Method "starting_point" not implemented')


class SeismicModel(Model):
    '''
    This class computes the misfit function and its gradient
    for a given model.
    Uses the same notation as in the paper.
    '''
    def __init__(self, azimuth, takeoff_angles, velocities, Ao = []):
        '''
        Initialize with the parts of the model that don't change.
        All angles are in RADIANS!!!
        
        Args:
        phi = raypath's azimuth
        i, j = P and S wave takeoff angles
        alpha, beta = P and S wave velocities
        Ao = observed amplitudes
        '''
        # extract inputs
        i, j = takeoff_angles
        i, j = np.deg2rad(i), np.deg2rad(j)
        alpha_h, beta_h = velocities
        
        # initialize variables for tracking
        self.misfits = []
        self.iterates = []
        self.optimal_iterates = []
        self.optimal_laplacians = []
        self.misfits = []
        self.tp_axes = []
        self.amplitudes = []
        self.optimal_amplitudes = []
        self.half_angles = []
        self.runs, self.converged = 0, 0
        self.filtered_outliers = False
        
        # initialize variables for hybrid search
        self.convergence_rates = []
        self.optimal_parameterizations = []
        self.optimal_errors = []
        self.optimal_axes = []
        self.sampled_amplitudes = []
        self.sampled_weights = []
        
        # initialize constant Jacobian matrix
        factors = np.array([alpha_h, beta_h, beta_h])**3
        self.J_A = np.array(
            [[(3*np.cos(i)**2 - 1), -np.sin(2*i), -np.sin(i)**2, 0, 0],
             [1.5*np.sin(2*j), np.cos(2*j), 0.5*np.sin(2*j), 0, 0],
             [0, 0, 0, np.cos(j), np.sin(j)]])
        
        # scale each row of J_A by the corresponding factor
        self.J_A /= factors[:,np.newaxis]
        
        # also sture J and nabla_E for later use
        self.J_eta = np.zeros((5,3))
        self.nabla_Phi = np.zeros(3)
        
        # store other inputs for later use
        self.phi = azimuth
        self.Ao = Ao
    
    
    def __call__(self, params, set_Ao = False):
        ''' Compute the value of the misfit function at params. '''
        # extract parameters
        psi, delta, lamb = params
        
        # compute intermediate variables
        sR = .5*np.sin(lamb) * np.sin(2*delta)
        qR = np.sin(lamb) * np.cos(2*delta) * np.sin(psi-self.phi) + \
            np.cos(lamb) * np.cos(delta) * np.cos(psi-self.phi)
        pR = np.cos(lamb) * np.sin(delta) * np.sin(2*(psi-self.phi)) - \
            .5*np.sin(lamb) * np.sin(2*delta) * np.cos(2*(psi-self.phi))
        qL = -np.cos(lamb) * np.cos(delta) * np.sin(psi-self.phi) + \
            np.sin(lamb) * np.cos(2*delta) * np.cos(psi-self.phi)
        pL = .5*np.sin(lamb) * np.sin(2*delta) * np.sin(2*(psi-self.phi)) + \
            np.cos(lamb) * np.sin(delta) * np.cos(2*(psi-self.phi))
        
        # compute and store A from eta
        eta = np.array([sR, qR, pR, qL, pL])
        self.A = self.J_A @ eta
        
        # set observed amplitudes for synthetic parameters
        if set_Ao: self.Ao = self.A
        
        # compute and store misfit
        Phi = -(fn.unit_vec(self.Ao) @ fn.unit_vec(self.A))
        
        return Phi
    
    
    def gradient(self, params, store=True):
        ''' Compute the gradient of the misfit function at params. '''
        psi, delta, lamb = params

        # update Jacobian J_eta
        rel_psi = psi - self.phi
        a, b, c = np.sin(lamb), np.cos(2*delta), np.cos(lamb)
        d, e, f = np.sin(2*delta), np.sin(delta), np.cos(delta)
        g, h, p = np.sin(rel_psi), np.cos(rel_psi), np.cos(2*rel_psi)
        q = np.sin(2*rel_psi)
        J_eta = np.array([[0, a*b, .5*c*d],
                   [a*b*h - c*f*g, -2*a*d*g - c*e*h, c*b*g - a*f*h],
                   [2*c*e*p + a*d*q, c*f*q - a*b*p, -a*e*q - .5*c*d*p],
                   [-c*f*h - a*b*g, -c*e*g - 2*a*d*h, a*f*g + c*b*h],
                   [a*d*p - 2*c*e*q, a*b*q + c*f*p, .5*c*d*q - a*e*p]])
        
        # update nabla_Phi
        Phi = self(params)
        nabla_Phi = -(Phi*fn.unit_vec(self.A) + fn.unit_vec(self.Ao))
        nabla_Phi /= linalg.norm(self.A)
        
        # compute the gradient and store its norm
        grad = J_eta.T @ self.J_A.T @ nabla_Phi
        
        # store the current misfit and iterate during gradient step
        if store:
            self.misfits.append(Phi)
            self.iterates.append(params)
            self.amplitudes.append(self.A)
        
        return grad
    
    
    def laplacian(self, params, h=1e-6):
        ''' Approximate the laplacian of the misfit function at params. '''
        grad_psi_plus = self.gradient(params + h*fn.i_hat, store=False)
        grad_psi_minus = self.gradient(params - h*fn.i_hat, store=False)
        grad_delta_plus = self.gradient(params + h*fn.j_hat, store=False)
        grad_delta_minus = self.gradient(params - h*fn.j_hat, store=False)
        grad_lamb_plus = self.gradient(params + h*fn.k_hat, store=False)
        grad_lamb_minus = self.gradient(params - h*fn.k_hat, store=False)
        
        laplacian_psi = (grad_psi_plus - grad_psi_minus) / (2*h)
        laplacian_psi = laplacian_psi @ fn.i_hat
        laplacian_delta = (grad_delta_plus - grad_delta_minus) / (2*h)
        laplacian_delta = laplacian_delta @ fn.j_hat
        laplacian_lamb = (grad_lamb_plus - grad_lamb_minus) / (2*h)
        laplacian_lamb = laplacian_lamb @ fn.k_hat
        laplacian = laplacian_psi + laplacian_delta + laplacian_lamb
        
        return laplacian
    
    
    def starting_point(self):
        ''' Return a starting point for the optimization algorithm. '''
        return np.zeros(3)
    
    
    def get_A(self):
        ''' Return the current value of A. '''
        return self.A
    
    
    def get_Ao(self):
        ''' Return the value of Ao. '''
        return self.Ao
    
    
    def set_Ao(self, Ao):
        ''' Set the observed amplitudes. '''
        self.Ao = Ao
    
    
    def update_convergence(self, converged):
        ''' Update the number of runs and the number of convergences. '''
        self.runs += 1
        self.converged += converged
    
    
    def convergence_rate(self):
        ''' Return the convergence rate. '''
        return 100*self.converged/self.runs
    
    
    def get_iterates(self):
        ''' Return parameter iterates from optimization. '''
        return self.iterates
    
    
    def get_misfits(self):
        ''' Return misfit values per iterate from optimizaiton. '''
        return self.misfits
    
    
    def set_optimal_iterate(self, params):
        ''' Set the optimal parameters. '''
        self.optimal_iterates.append(params)
    
    
    def set_optimal_laplacian(self, params):
        ''' Set the laplacian. '''
        laplacian = self.laplacian(params)
        self.optimal_laplacians.append(laplacian)
    
    
    def set_optimal_amplitude(self):
        ''' Set the amplitude vector. '''
        self.optimal_amplitudes.append(self.A)
        
    
    def get_optimal_iterates(self):
        ''' Return the optimal parameters. '''
        return self.optimal_iterates
    
    
    def get_optimal_amplitudes(self):
        ''' Return the optimal amplitudes. '''
        return self.optimal_amplitudes
    
    
    def get_tp_axes(self):
        ''' Return the tp axes. '''
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        return self.tp_axes
    
    
    def get_half_angles(self):
        ''' Return the half angles. '''
        return self.half_angles
    
    
    def get_central_tps(self):
        ''' Return the central tps for the optimal tp axes. '''
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        central, pos = fn.regression_axes(self.tp_axes)
        output = np.zeros_like(self.tp_axes)
        
        for i in range(len(self.tp_axes)):
            tp = self.tp_axes[i]
            to_project = tp[pos]
            projected = fn.starting_direc(central, to_project)
            to_insert = [central, projected] if pos == 0 else [projected, central]
            output[i] = to_insert
        
        return output
    
    
    def filter_outliers(self, threshold=2):
        ''' Filter out outliers based on Z-score of optimal amplitude norms. '''
        norms = np.array([linalg.norm(a) for a in self.optimal_amplitudes])
        z_scores = np.abs((norms - np.mean(norms)) / np.std(norms))
        self.optimal_iterates = [m for i, m in enumerate(self.optimal_iterates) 
                                 if z_scores[i] < threshold]
        self.optimal_amplitudes = [a for i, a in enumerate(self.optimal_amplitudes) 
                                   if z_scores[i] < threshold]
        self.optimal_laplacians = [l for i, l in enumerate(self.optimal_laplacians)
                                      if z_scores[i] < threshold]
        self.filtered_outliers = True
    
    
    def reset(self):
        ''' Reset the misfits and iterates. '''
        self.misfits = []
        self.iterates = []
        self.optimal_iterates = []
        self.optimal_laplacians = []
        self.misfits = []
        self.tp_axes = []
        self.amplitudes = []
        self.optimal_amplitudes = []
        self.half_angles = []
        self.runs, self.converged = 0, 0
        self.filtered_outliers = False
        
    
    def mirror(self, what, index=0):
        '''
        Mirror the optimal parameters to show only one fault plane.
        '''
        if not self.filtered_outliers: self.filter_outliers()
        
        if len(what) == 1:
            what = what[0]
            if what == 'optimals':
                optimal_iterates = []
                for m in self.optimal_iterates:
                    t, p = fn.sdr2tp(m)
                    optimal_iterates.append(fn.tp2sdr(t, p)[index])
                self.optimal_iterates = optimal_iterates
            elif what == 'iterates':
                iterates = []
                for m in self.iterates:
                    t, p = fn.sdr2tp(m)
                    iterates.append(fn.tp2sdr(t, p)[index])
                self.iterates = iterates
            elif what == 'axes':
                tp_axes = []
                for m in self.optimal_iterates:
                    t, p = fn.sdr2tp(m)
                    tp_axes.append([t, p])
                self.tp_axes = fn.filter_axes(tp_axes)
                assert len(self.tp_axes) == len(self.optimal_iterates), \
                    'Not as many tp axes as optimal iterates'
            else:
                raise ValueError('Invalid keyword for "what"')
        elif len(what) > 1:
            for w in what:
                self.mirror([w], index)
        else:
            raise ValueError('Invalid argument for "what"')
    
    
    def plot_misfit(self):
        '''
        Plot the misfits in a subplot.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(self.misfits)
        ax.set_title('Misfit function')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Misfit')
        plt.show()
        
    
    def plot_half_angles(self, bins=20):
        '''
        Plot the half angles in a histogram.
        This is a diagnostic plot.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(np.rad2deg(self.half_angles), bins=bins)
        ax.set_title('Diagnostic histogram of half-angles')
        ax.set_xlabel('Half angle (deg)')
        ax.set_ylabel('Frequency')
        plt.show()
        
        
    def plot_iterates_2D(self, cmap='rainbow', s=10, optimal=True, index=2):
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
                self.mirror(['optimals', 'iterates'], index)
                iterates = self.iterates
                optimal_iterates = self.optimal_iterates
            else:
                self.mirror(['optimals'], index)
                optimal_iterates = self.optimal_iterates
        elif index == 2:
            iterates, optimal_iterates = [], []
            if not optimal:
                self.mirror(['optimals', 'iterates'], 0)
                iterates.extend(self.iterates)
                optimal_iterates.extend(self.optimal_iterates)
                self.mirror(['optimals', 'iterates'], 1)
                iterates.extend(self.iterates)
                optimal_iterates.extend(self.optimal_iterates)
            else:
                self.mirror(['optimals'], 0)
                optimal_iterates.extend(self.optimal_iterates)
                self.mirror(['optimals'], 1)
                optimal_iterates.extend(self.optimal_iterates)
        
        opt_strikes = [np.rad2deg(m[0]) for m in optimal_iterates]
        opt_dips = [np.rad2deg(m[1]) for m in optimal_iterates]
        opt_rakes = [np.rad2deg(m[2]) for m in optimal_iterates]
        
        # convert the angles to degrees
        if not optimal:
            strikes = [np.rad2deg(m[0]) for m in iterates]
            dips = [np.rad2deg(m[1]) for m in iterates]
            rakes = [np.rad2deg(m[2]) for m in iterates]
            weights = -np.array(self.misfits)
        else:
            weights = np.array(self.optimal_laplacians)
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
        
        
    def plot_amplitudes(self, elev=30, azim=45, cmap='rainbow', s=10, alpha=0.5, iterates=False,
                        observed=True):
        '''
        Make a 3D scatter plot of the optimal amplitudes.
        '''
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        opt_AP = [a[0] for a in self.optimal_amplitudes]
        opt_ASV = [a[1] for a in self.optimal_amplitudes]
        opt_ASH = [a[2] for a in self.optimal_amplitudes]
        
        if iterates:
            AP = [a[0] for a in self.amplitudes]
            ASV = [a[1] for a in self.amplitudes]
            ASH = [a[2] for a in self.amplitudes]
            weights = -np.array(self.misfits)
            
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
                self.Ao[0], self.Ao[1], self.Ao[2],
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
    
    
    def plot_tp_axes(self, elev=30, azim=45, half=False, central=False):
        '''
        Make a 3D plot of the optimal tp axes.
        '''
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        if central: to_plot = self.get_central_tps()
        else: to_plot = self.tp_axes
        zero = np.zeros(3)
        
        # compute the central axis
        c, _ = fn.regression_axes(self.tp_axes)
        
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
                c='green', alpha=0.5, label="central", linewidth=3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.title('Optimal tp axes', fontsize=15)
        
        # adjust the view angle
        ax.view_init(elev=elev, azim=azim)
        
        plt.show()
        
    
    def plot_beachballs(self, central=False, order_by='strike',width=10, max_plot=50,
                        facecolor='blue'):
        '''
        Plot beachballs for the optimal solutions.
        '''
        if not self.filtered_outliers: self.filter_outliers()
        og_set = self.optimal_iterates
        if central:
            solution_set = [fn.tp2sdr(t, p)[0] for t, p in self.get_central_tps()]
            facecolor = 'red'
        else: solution_set = self.optimal_iterates
        
        assert len(solution_set) == len(og_set), f'Length mismatch: {len(solution_set)} vs {len(og_set)}'
        # get sorting order from og_set
        if order_by == 'strike': order = np.argsort([s[0] for s in og_set])
        elif order_by == 'dip': order = np.argsort([s[1] for s in og_set])
        elif order_by == 'rake': order = np.argsort([s[2] for s in og_set])
        
        # now sort solution_set
        solution_set = [solution_set[i] for i in order]
        
        
        bp.grid_beach(solution_set, width, max_plot, facecolor)
        
    
    def laplacian_flow(self):
        ''' Return the direction opposing largest laplacian. '''
        # index of max laplacian
        max_index = np.argmax(self.optimal_laplacians)
        
        # get tp axes
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        
        # get central axis
        central, axis = fn.regression_axes(self.tp_axes)
        
        # get the optimal axis
        optimal = self.tp_axes[max_index][axis]
        
        # turn into spherical coordinates
        flow = fn.rect2pol(optimal) - fn.rect2pol(central)
        
        return flow
    
    
    def optimal_parameterization(self):
        '''
        Compute optimal parameterization of fault planes.
        Returns axis direction, half-angle, error and name.
        '''
        if not self.filtered_outliers: self.filter_outliers()
        
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        central, position = fn.regression_axes(self.tp_axes)
        
        for axis in self.tp_axes:
            half_angle = np.arccos(axis[position] @ central)
            self.half_angles.append(half_angle)
        half_angle = np.mean(self.half_angles)
        error = np.std(self.half_angles)
        
        if central[2] < 0: central *= -1
        
        rho, theta, phi = fn.rect2pol(central)
        assert np.abs(rho - 1) < fn.eps, 'Central axis is not a unit vector'
        params = np.array([theta, phi, half_angle])
        
        return params, error, position
    
    
    ######################################################################
    # (MONTE-CARLO) METHODS FOR UNCERTAINTY QUANTIFICATION
    ######################################################################
    
    
    def super_reset(self):
        ''' Reset all variables. '''
        self.reset()
        self.convergence_rates = []
        self.optimal_parameterizations = []
        self.optimal_errors = []
        self.optimal_axes = []
        self.sampled_amplitudes = []
        self.sampled_weights = []
    
    
    def init_uncertainty(self):
        ''' Initialize the uncertainty ellipsoid. '''
        self.set_Uo()
        self.set_chi2()
    
    
    def set_Uo(self, Uo = []):
        ''' Set the observed uncertainties and chi2 value. '''
        if len(Uo) == 0: Uo = np.abs(self.Ao*0.01)
        self.Uo = Uo
    
    
    def get_Uo(self):
        ''' Return the observed uncertainties. '''
        return self.Uo


    def set_chi2(self, chi2 = 1):
        ''' Set the chi2 value. '''
        self.chi2 = chi2
    
    
    def get_chi2(self):
        ''' Return the chi2 value. '''
        return self.chi2


    def get_normal(self):
        ''' Return the normal vector. '''
        assert len(self.Uo) > 0, 'Observed uncertainties not set'
        Sigma_tilde = np.diag(self.Uo)/self.chi2
        Sigma_tilde_inv = linalg.inv(Sigma_tilde)
        normal = Sigma_tilde_inv @ self.Ao
        
        return fn.unit_vec(normal)
    
    
    def sample_amplitudes(self, dd=15, num_samples=50):
        ''' Sample amplitudes in a systematic elliptical cone. '''
        normal = self.get_normal()
        Ao = self.get_Ao()
        Uo = self.get_Uo()
        chi2 = self.get_chi2()
        self.sampled_amplitudes = fn.systematic_ellipse(normal, Ao, Uo, chi2, dd, num_samples)
        self.sampled_weights = [fn.unit_vec(As) @ fn.unit_vec(Ao) for As in self.sampled_amplitudes]
        
    
    def get_sampled_amplitudes(self):
        ''' Return the sampled amplitudes. '''
        return self.sampled_amplitudes
    
    
    def plot_sampled_amplitudes(self, cmap='rainbow', s=10, alpha=1, azimuth=45, elevation=30):
        '''
        Make a 3D scatter plot of the sampled amplitudes.
        Weight them by cosine similarity with Ao.
        '''
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        AP = [As[0] for As in self.sampled_amplitudes]
        ASV = [As[1] for As in self.sampled_amplitudes]
        ASH = [As[2] for As in self.sampled_amplitudes]
        weights = np.array(self.sampled_weights)
        
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
    
     
    def hybrid_search(self, num_runs=30, print_every=0):
        '''
        Perform a random hybrid search to find the optimal parameterization.
        '''
        self.reset()
        config = opt.get_config()
        starts = fn.random_params(num_runs)
        
        for index, start in enumerate(starts):
            if print_every > 0 and index % print_every == 0:
                print(f'Run {index} of {num_runs}')
            opt.minimize(self, config, start)
            
        self.convergence_rates.append(self.convergence_rate())
        params, error, position = self.optimal_parameterization()
        self.optimal_parameterizations.append(params)
        self.optimal_errors.append(error)
        self.optimal_axes.append(position)
    
    
    def get_convergence_rates(self):
        ''' Return the convergence rates. '''
        return self.convergence_rates
    
    
    def get_optimal_parameterizations(self):
        ''' Return the optimal parameterizations. '''
        return self.optimal_parameterizations
    
    
    def get_optimal_errors(self):
        ''' Return the optimal errors. '''
        return self.optimal_errors
    
    
    def get_optimal_axes(self):
        ''' Return the optimal axes. '''
        return self.optimal_axes
    
    
    def post_filter(self, threshold=90, orig=False):
        '''
        Remove the minority axis from the optimal parameterizations. 
        Filter out errors above a threshold in DEGREES.
        threshold=0 means revert to the original values.
        '''
        threshold = np.deg2rad(threshold)
        majority_axis = int(self.optimal_axes.count(1) > self.optimal_axes.count(0))
        
        # save original values
        if orig:
            self.orig_optimal_parameterizations = self.optimal_parameterizations
            self.orig_optimal_errors = self.optimal_errors
            self.orig_convergence_rates = self.convergence_rates
            self.orig_sampled_weights = self.sampled_weights
            self.orig_sampled_amplitudes = self.sampled_amplitudes
            self.orig_optimal_axes = self.optimal_axes
        
        if threshold == 0:
            self.optimal_parameterizations = self.orig_optimal_parameterizations
            self.optimal_errors = self.orig_optimal_errors
            self.convergence_rates = self.orig_convergence_rates
            self.sampled_weights = self.orig_sampled_weights
            self.sampled_amplitudes = self.orig_sampled_amplitudes
            self.optimal_axes = self.orig_optimal_axes
            return
    
        # filter out the minority axis
        self.optimal_parameterizations = [p for i, p in enumerate(self.optimal_parameterizations)
                                            if self.optimal_axes[i] == majority_axis]
        self.optimal_errors = [e for i, e in enumerate(self.optimal_errors)
                                if self.optimal_axes[i] == majority_axis]
        self.convergence_rates = [c for i, c in enumerate(self.convergence_rates)
                                    if self.optimal_axes[i] == majority_axis]
        self.sampled_weights = [w for i, w in enumerate(self.sampled_weights)
                                    if self.optimal_axes[i] == majority_axis]
        self.sampled_amplitudes = [a for i, a in enumerate(self.sampled_amplitudes)
                                    if self.optimal_axes[i] == majority_axis]
        self.optimal_axes = [a for a in self.optimal_axes if a == majority_axis]
        
        # filter out errors above threshold
        self.optimal_parameterizations = [p for i, p in enumerate(self.optimal_parameterizations)
                                            if self.optimal_errors[i] < threshold]
        self.sampled_weights = [w for i, w in enumerate(self.sampled_weights)
                                    if self.optimal_errors[i] < threshold]
        self.sampled_amplitudes = [a for i, a in enumerate(self.sampled_amplitudes)
                                    if self.optimal_errors[i] < threshold]
        self.convergence_rates = [c for i, c in enumerate(self.convergence_rates)
                                        if self.optimal_errors[i] < threshold]
        self.optimal_axes = [a for i, a in enumerate(self.optimal_axes)
                                if self.optimal_errors[i] < threshold]
        self.optimal_errors = [e for e in self.optimal_errors if e < threshold]

    
    def monte_carlo(self, dd=15, num_samples=50, num_runs=30, print_every=0):
        '''
        Trace out shape of the uncertainty ellipsoid in the parameter space.
        '''
        self.super_reset()
        
        self.sample_amplitudes(dd, num_samples)
        Ao = self.get_Ao()
        num_samples = len(self.sampled_amplitudes)
        
        for i, As in enumerate(self.sampled_amplitudes):
            print(f"Sample {i} of {num_samples}")
            self.set_Ao(As)
            self.hybrid_search(num_runs, print_every)
        
        self.set_Ao(Ao)
        self.post_filter(orig=True)
    
    
    def plot_uncertainty_2D(self, cmap='rainbow', s=10, scale=0.5):
        '''
        Plot the uncertainty ellipsoid in the parameter space.
        '''
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        thetas = [np.rad2deg(p[0]) for p in self.optimal_parameterizations]
        phis = [np.rad2deg(p[1]) for p in self.optimal_parameterizations]
        half_angles = [np.rad2deg(p[2]) for p in self.optimal_parameterizations]
        weights = np.array(self.sampled_weights)
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
            theta_flow, phi_flow = scale*fn.unit_vec(self.laplacian_flow()[1:])
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
    
    
    def plot_uncertainty_3D(self, elev=30, azim=45, cmap='rainbow', s=10, alpha=1):
        '''
        Plot the uncertainty ellipsoid in the parameter space.
        '''
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        thetas = [p[0] for p in self.optimal_parameterizations]
        phis = [p[1] for p in self.optimal_parameterizations]
        half_angles = [p[2] for p in self.optimal_parameterizations]
        weights = np.array(self.sampled_weights)
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
        
        
    def plot_optimal_errors(self, bins=10):
        '''
        Plot the optimal errors in a histogram.
        This is a diagnostic plot.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(np.rad2deg(self.optimal_errors), bins=bins)
        ax.set_title('Diagnostic histogram of optimal errors')
        ax.set_xlabel('Error (deg)')
        ax.set_ylabel('Frequency')
        
        plt.show()


# TODO: Make SeismicModel a parent class to allow for other cost functions
# TODO: Will need to make optimal configs in optimizer per cost function


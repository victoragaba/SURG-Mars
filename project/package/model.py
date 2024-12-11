'''
Name: Victor Agaba

Date: 2nd November 2024

The goal of this module is to create a model of the nonlinear inverse
problem to which we will apply the optimization algorithms.
'''


import numpy as np
from numpy import linalg
from . import functions as fn
from . import optimizer as opt


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
    
    
    #
    # optimization methods
    #
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
    
    
    #
    # setters for one run
    #
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
    
    
    def set_Ao(self, Ao):
        ''' Set the observed amplitudes. '''
        self.Ao = Ao
    
    
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
    
    
    def update_convergence(self, converged):
        ''' Update the number of runs and the number of convergences. '''
        self.runs += 1
        self.converged += converged
    
    
    #
    # getters for one run
    #
    def get_A(self):
        ''' Return the current value of A. '''
        return self.A
    
    
    def get_Ao(self):
        ''' Return the value of Ao. '''
        return self.Ao
    
    
    def get_iterates(self):
        ''' Return parameter iterates from optimization. '''
        return self.iterates
    
    
    def get_misfits(self):
        ''' Return misfit values per iterate from optimizaiton. '''
        return self.misfits
    
    
    def get_amplitudes(self):
        ''' Return the amplitudes per iterate from optimization. '''
        return self.amplitudes
    
    
    def get_convergence_rate(self):
        ''' Return the convergence rate. '''
        return 100*self.converged/self.runs
    
    
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
    
    
    def get_optimal_laplacians(self):
        ''' Return the optimal laplacians. '''
        return self.optimal_laplacians
    
    
    def get_central_tps(self):
        ''' Return the central tps for the optimal tp axes. '''
        if len(self.tp_axes) == 0: self.mirror(['axes'])
        central, pos = fn.regression_axes(self.tp_axes)
        central_tps = np.zeros_like(self.tp_axes)
        
        for i in range(len(self.tp_axes)):
            tp = self.tp_axes[i]
            to_project = tp[pos]
            projected = fn.starting_direc(central, to_project)
            to_insert = [central, projected] if pos == 0 else [projected, central]
            central_tps[i] = to_insert
        
        return central_tps
    
    
    def get_laplacian_flow(self):
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
    
    
    def get_optimal_parameterization(self, threshold=2):
        '''
        Compute optimal parameterization of fault planes.
        Returns axis direction, half-angle, error and name.
        '''
        if not self.filtered_outliers: self.filter_outliers(threshold)
        
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
    
    
    #
    # pre and post processing for one run
    #
    def filter_outliers(self, threshold=2):
        ''' Filter out outliers based on Z-score of optimal amplitude norms. '''
        if self.filtered_outliers: return
        norms = np.array([linalg.norm(a) for a in self.optimal_amplitudes])
        z_scores = np.abs((norms - np.mean(norms)) / np.std(norms))
        self.optimal_iterates = [m for i, m in enumerate(self.optimal_iterates) 
                                 if z_scores[i] < threshold]
        self.optimal_amplitudes = [a for i, a in enumerate(self.optimal_amplitudes) 
                                   if z_scores[i] < threshold]
        self.optimal_laplacians = [l for i, l in enumerate(self.optimal_laplacians)
                                      if z_scores[i] < threshold]
        self.filtered_outliers = True
        
    
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
        
        
    def hybrid_search(self, num_runs=30, print_every=0, threshold=2):
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
            
        self.convergence_rates.append(self.get_convergence_rate())
        params, error, position = self.get_optimal_parameterization(threshold)
        self.optimal_parameterizations.append(params)
        self.optimal_errors.append(error)
        self.optimal_axes.append(position)
    
    
    ######################################################################
    # (MONTE-CARLO) METHODS FOR UNCERTAINTY QUANTIFICATION
    ######################################################################
    
    
    #
    # setters for many runs
    #
    def super_reset(self):
        ''' Reset all variables. '''
        self.reset()
        self.convergence_rates = []
        self.optimal_parameterizations = []
        self.optimal_errors = []
        self.optimal_axes = []
        self.sampled_amplitudes = []
        self.sampled_weights = []
    
    
    def set_Uo(self, Uo = []):
        ''' Set the observed uncertainties and chi2 value. '''
        if len(Uo) == 0: Uo = np.abs(self.Ao*0.01)
        self.Uo = Uo
        
    
    def set_chi2(self, chi2 = 1):
        ''' Set the chi2 value. '''
        self.chi2 = chi2
    
    
    def init_uncertainty(self):
        ''' Initialize the uncertainty ellipsoid. '''
        self.set_Uo()
        self.set_chi2()
    
    
    def sample_amplitudes(self, dd=15, num_samples=50):
        ''' Sample amplitudes in a systematic elliptical cone. '''
        normal = self.get_normal()
        Ao = self.get_Ao()
        Uo = self.get_Uo()
        chi2 = self.get_chi2()
        self.sampled_amplitudes = fn.systematic_ellipse(normal, Ao, Uo, chi2, dd, num_samples)
        self.sampled_weights = [fn.unit_vec(As) @ fn.unit_vec(Ao) for As in self.sampled_amplitudes]
    
    
    #
    # getters for many runs
    #
    def get_Uo(self):
        ''' Return the observed uncertainties. '''
        return self.Uo
    
    
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
    
    
    def get_sampled_amplitudes(self):
        ''' Return the sampled amplitudes. '''
        return self.sampled_amplitudes
    
    
    def get_sampled_weights(self):  
        ''' Return the sampled weights. '''
        return self.sampled_weights
    
     
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
    
    
    #
    # pre and post processing for many runs
    #
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
    
        # filter out minority axis
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

    
    def monte_carlo(self, dd=15, num_samples=50, num_runs=30, print_every=0, threshold=2):
        '''
        Trace out shape of the uncertainty ellipsoid in the parameter space.
        '''
        self.super_reset()
        
        self.sample_amplitudes(dd, num_samples)
        Ao = self.Ao
        num_samples = len(self.sampled_amplitudes)
        
        for i, As in enumerate(self.sampled_amplitudes):
            print(f"Sample {i} of {num_samples}")
            self.set_Ao(As)
            self.hybrid_search(num_runs, print_every, threshold)
        
        self.set_Ao(Ao)
        self.post_filter(orig=True)


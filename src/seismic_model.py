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


class Model:
    '''
    Base class for an unconstrained optimzation problem.
    Catch anything whose method is not implemented.
    '''
    
    def __call__(self, params):
        raise Exception('Class is not callable')

    def gradient(self, params):
        raise Exception('Method "gradient" is not implemented')

    def hessian(self, params):
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
        
        # initialize misfits iterates for tracking
        self.misfits = []
        self.iterates = []
        
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
        
        # optionally set the observed amplitudes
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
        
        return grad
    
    
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
    
    
    def get_iterates(self):
        ''' Return parameter iterates from optimization. '''
        return self.iterates
    
    
    def get_misfits(self):
        ''' Return misfit values per iterate from optimizaiton. '''
        return self.misfits
    
    def reset(self):
        ''' Reset the misfits and iterates. '''
        self.misfits = []
        self.iterates = []
    
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
    
    def plot_iterates_2D(self, cmap='rainbow', s=10):
        '''
        Make 3 2D plots of the iterates: psi, delta, lambda.
        psi against delta, psi against lambda, delta against lambda.
        Join the points with a line and color them by iteration number.
        '''
        
        strikes = [np.rad2deg(m[0]) for m in self.iterates]
        dips = [np.rad2deg(m[1]) for m in self.iterates]
        rakes = [np.rad2deg(m[2]) for m in self.iterates]
        weights = -np.array(self.misfits)
        
        # make the plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.scatter(strikes, dips, c=weights, cmap=cmap, s=s)
        # ax1.plot(strikes, dips, c='k', alpha=0.5, linewidth=0.5)
        ax1.set_title('Strike against Dip')
        ax1.set_xlabel('Strike (deg)')
        ax1.set_ylabel('Dip (deg)')
        ax2.scatter(strikes, rakes, c=weights, cmap=cmap, s=s)
        # ax2.plot(strikes, rakes, c='k', alpha=0.5, linewidth=0.5)
        ax2.set_title('Strike against Rake')
        ax2.set_xlabel('Strike (deg)')
        ax2.set_ylabel('Rake (deg)')
        ax3.scatter(dips, rakes, c=weights, cmap=cmap, s=s)
        # ax3.plot(dips, rakes, c='k', alpha=0.5, linewidth=0.5)
        ax3.set_title('Dip against Rake')
        ax3.set_xlabel('Dip (deg)')
        ax3.set_ylabel('Rake (deg)')
        
        # add a colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
        cbar.set_label('Cosine similarity')
        
        plt.show()
        
        
    def plot_iterates_3D(self, elev=30, azim=45, cmap='rainbow', s=10):
        '''
        Make a 3D scatter plot of the iterates (psi, delta, lambda) and project
        them onto the 2D planes: psi-delta, psi-lambda, delta-lambda.
        Color the points by iteration number.
        '''
        
        # convert the angles to deg
        strikes = [np.rad2deg(m[0]) for m in self.iterates]
        dips = [np.rad2deg(m[1]) for m in self.iterates]
        rakes = [np.rad2deg(m[2]) for m in self.iterates]
        weights = -np.array(self.misfits)
        
        # create a 3D plot
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111, projection='3d')
        plt.suptitle('3D visualization of the iterates', fontsize=15)
        scatter = ax.scatter(strikes, dips, rakes, c=weights, cmap=cmap, s=s)
        ax.set_xlabel('Strike (deg)')
        ax.set_ylabel('Dip (deg)')
        ax.set_zlabel('Rake (deg)')
        
        # adjust the view
        ax.view_init(elev=elev, azim=azim)
        
        # add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Cosine similarity')
        
        plt.show()

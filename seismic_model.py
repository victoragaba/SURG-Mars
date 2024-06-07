'''
Name: Victor Agaba

Date: 5th December 2023

The goal of this file is to create a model of the nonlinear inverse
problem to which we will apply the optimization algorithms.
'''


from numpy import *
import matplotlib.pyplot as plt


class InvProblem:
    '''
    Base class for an unconstrained optimzation problem.
    Catch anything whose method is not implemented.
    '''
    def __init__(self):
        pass

    def value(self, m):
        raise Exception('Method "value" is not implemented')

    def gradient(self, m):
        raise Exception('Method "gradient" is not implemented')

    def hessian(self, m):
        raise Exception('Method "hessian" not is implemented')

    def starting_point(self):
        raise Exception('Method "starting_point" not implemented')


class SeismicModel(InvProblem):
    '''
    This class will compute the misfit function and its gradient
    for a given model.
    '''
    
    def __init__(self, phi, i, j, alpha, beta, d):
        '''
        Initialize with the parts of the model that don't change.
        Uses the same notation as in the project description.
        ALL ANGLES ARE IN RADIANS!!!
        '''
        # initialize misfits and grad norms for tracking
        self.misfits, self.grad_norms, self.iterates = [], [], []
        
        # initialize constant Jacobian matrix
        factors = array([alpha, beta, beta])**3
        self.C = array([[(3*cos(i)**2 - 1), -sin(2*i), -sin(i)**2, 0, 0],
                      [1.5*sin(2*j), cos(2*j), 0.5*sin(2*j), 0, 0],
                      [0, 0, 0, cos(j), sin(j)]])
        
        # scale each row of C by the corresponding factor
        self.C /= factors[:,newaxis]
        
        # also sture J and nabla_E for later use
        self.J = zeros((5,3))
        self.nabla_E = zeros(3)
        
        # store other inputs for later use
        self.phi = phi
        self.d = d
        
        # store current value of m, g and E
        self.m, self.g, self.E = None, None, None
    
    
    def value(self, m):
        '''
        Compute the value of the misfit function at m.
        This works a lot like Rpattern from the summer project.
        '''
        self.m = m
        psi, delta, lamb = m
        
        # from Maddy's paper
        sR = .5*sin(lamb)*sin(2*delta)
        qR = sin(lamb)*cos(2*delta)*sin(psi-self.phi) + \
            cos(lamb)*cos(delta)*cos(psi-self.phi)
        pR = cos(lamb)*sin(delta)*sin(2*(psi-self.phi)) - \
            .5*sin(lamb)*sin(2*delta)*cos(2*(psi-self.phi))
        qL = -cos(lamb)*cos(delta)*sin(psi-self.phi) + \
            sin(lamb)*cos(2*delta)*cos(psi-self.phi)
        pL = .5*sin(lamb)*sin(2*delta)*sin(2*(psi-self.phi)) + \
            cos(lamb)*sin(delta)*cos(2*(psi-self.phi))
        
        # compute g from vector h
        h = array([sR, qR, pR, qL, pL])
        self.g = self.C @ h
        
        # compute the misfit function: negative cosine similarity
        self.E = -dot(self.d, self.g)/(linalg.norm(self.d)*linalg.norm(self.g))
        
        return self.E
        
    
    def gradient(self, m):
        '''
        Compute the gradient of the misfit function at m.
        '''
        psi, delta, lamb = m

        # update Jacobian J
        a, b, c, d = sin(lamb), cos(2*delta), cos(lamb), sin(2*delta)
        e, f, g, h = cos(psi-self.phi), cos(delta), sin(psi-self.phi), sin(delta)
        p, q = cos(2*(psi-self.phi)), sin(2*(psi-self.phi))
        self.J = array([[0, a*b, .5*c*d],
                   [a*b*e - c*f*g, -2*a*d*g - c*h*e, c*b*g - a*f*e],
                   [2*c*h*p + a*d*q, c*f*q - a*b*p, -a*h*q - .5*c*d*p],
                   [-c*f*e - a*b*g, -c*h*g - 2*a*d*e, a*f*g + c*b*e],
                   [a*d*p - 2*c*h*q, a*b*q + c*f*p, .5*c*d*q - a*h*p]])
        
        # update nabla_E
        cossim = -self.E
        ghat = self.g/linalg.norm(self.g)
        dhat = self.d/linalg.norm(self.d)
        self.nabla_E = (1/linalg.norm(self.g))*(cossim*ghat - dhat)
        
        # compute the gradient and store its norm
        grad = self.J.T @ self.C.T @ self.nabla_E
        self.grad_norms.append(linalg.norm(grad))
        
        # store the current misfit and iterate during gradient step
        self.misfits.append(self.E)
        self.iterates.append(self.m)
        
        return grad
    
    
    def set_initial_guess(self, sdr):
        '''
        Set the initial guess for the optimization algorithm.
        '''
        self.m = sdr
    
    
    def starting_point(self):
        '''
        Return a starting point for the optimization algorithm.
        '''
        return zeros(3)
    
    
    def get_g(self):
        '''
        Return the current value of g.
        '''
        return self.g
    
    
    def reset(self):
        '''
        Reset the misfits, gradient norms and iterates.
        '''
        self.misfits, self.grad_norms, self.iterates = [], [], []
        self.m, self.g, self.E = None, None, None
    
    
    def plot_misfit(self):
        '''
        Plot the misfits and gradient norms in subplots.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.misfits)
        ax1.set_title('Misfit function')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Misfit')
        ax2.plot(self.grad_norms)
        ax2.set_title('Gradient norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient norm')
        plt.show()
    
    def plot_iterates(self):
        '''
        Make 3 2D plots of the iterates: psi, delta, lambda.
        psi against delta, psi against lambda, delta against lambda.
        Join the points with a line and color them by iteration number.
        '''
        
        strikes = [rad2deg(m[0]) for m in self.iterates]
        dips = [rad2deg(m[1]) for m in self.iterates]
        rakes = [rad2deg(m[2]) for m in self.iterates]
        weights = -array(self.misfits)
        
        # make the plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle('Visualization of the iterates', fontsize=15)
        ax1.scatter(strikes, dips, c=weights, cmap='rainbow')
        ax1.plot(strikes, dips, c='k', alpha=0.5, linewidth=0.5)
        ax1.set_title('Strike against Dip')
        ax1.set_xlabel('Strike (degrees)')
        ax1.set_ylabel('Dip (degrees)')
        ax2.scatter(strikes, rakes, c=weights, cmap='rainbow')
        ax2.plot(strikes, rakes, c='k', alpha=0.5, linewidth=0.5)
        ax2.set_title('Strike against Rake')
        ax2.set_xlabel('Strike (degrees)')
        ax2.set_ylabel('Rake (degrees)')
        ax3.scatter(dips, rakes, c=weights, cmap='rainbow')
        ax3.plot(dips, rakes, c='k', alpha=0.5, linewidth=0.5)
        ax3.set_title('Dip against Rake')
        ax3.set_xlabel('Dip (degrees)')
        ax3.set_ylabel('Rake (degrees)')
        
        # add a colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(cmap='rainbow'), cax=cbar_ax)
        
        plt.show()

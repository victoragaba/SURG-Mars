# every necessary function for the paper is defined here

import numpy as np
from numpy import linalg

tol = 1e-6

def unit_vector(vector: list):
    """ Returns the unit vector of a vector. """
    return vector / linalg.norm(vector)


def boundary_points(Ao: list, Uo: list, As: list = [], chi2: float = 1):
    """ Returns the boundary points of the model.

    Args:
        Ao (list): Vector of observed amplitudes.
        Uo (list): Vector of measured uncertainties.
        As (list, optional): Vector of simulated amplitudes. Defaults to [].
        chi2 (float, optional): _description_. Defaults to 1.
    """
    if len(As) == 0:
        As = np.zeros(len(Ao))
        As[0] = 1
        if True: # TODO:...
            As[1] = 1
            
    
    # adjusted covariance matrix
    assert all(Uo > 0), "All uncertainties must be positive."
    Sigma = np.diag(Uo)/chi2
    Sigma_inv = linalg.inv(Sigma)
    
    # constants
    c1 = Ao@Sigma_inv@Ao
    c2 = Ao@Sigma_inv@As
    c3 = As@Sigma_inv@As
    
    # conditions
    assert c1 >= 1, "COnfidence ellipsoid is too big."
    assert c1*c3 > c2**2, "Cauchy-Schwarz inequality is not satisfied."
    
    # boundary points
    left = (1 - 1/c1)*Ao
    right = (1/c1)*np.sqrt((c1-1)/(c1*c3 - c2**2))*(c1*As - c2*Ao)
    Ab_low, Ab_high = left - right, left + right
    
    return Ab_low, Ab_high
    

def get_epsilon(Ao: list, bounds: tuple):
    """ Returns the epsilon value for the boundary points.

    Args:
        Ao (list): Vector of observed amplitudes.
        bounds (tuple): Tuple of boundary points.
    """
    pass


def main():
    # boundary_points(np.zeros(3), np.zeros(3), np.zeros(3))
    pass


if __name__ == '__main__':
    main()
    
# every necessary function for the paper is defined here

import numpy as np
from numpy import linalg


def unit_vector(vector: list):
    """ Returns the unit vector of a vector. """
    return vector / linalg.norm(vector)


def boundary_points(Ao: list, Uo: list, As: list = [], chi2: float = 1):
    """ Returns the boundary points of the model.

    Args:
        Ao (list): Vector of observed amplitudes.
        Uo (list): Vector of measured uncertainties.
        As (list, optional): _description_. Defaults to [].
        chi2 (float, optional): _description_. Defaults to 1.
    """
    if len(As) == 0:
        As = np.zeros(len(Ao))
        As[0] = 1
    
    Sigma = np.diag(Uo)/chi2
    Sigma_inv = linalg.inv(Sigma)
    
        


def main():
    # boundary_points(np.zeros(3), np.zeros(3), np.zeros(3))
    pass


if __name__ == '__main__':
    main()
    
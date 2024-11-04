'''
Name: Victor Agaba

Date: 4th November 2024

The goal of this module is to design optimization algorithms to be used
for inversion of seismic data.
'''


import numpy as np
from numpy import linalg


def get_config():
    '''
    Get the default configuration for the optimization algorithm.
    
    Output:
        config (dict): Default configuration for the optimization algorithm.
        method (str): Optimization method to use: 'SD', 'Newton', 'BFGS'.
        c_decrease (float): Sufficient decrease factor.
        c_increase (float): Curvature condition factor.
        k_max (int): Maximum number of iterations.
        alpha (float): Initial step size.
        rho (float): Backtracking line search parameter.
        tolerance (float): Stopping criterion.
        print_every (int): Print output every n iterations.
    '''
    config = {
        'method': 'SD',
        'c_decrease':1e-4,
        'c_increase': 1-1e-4,
        'k_max': 20000,
        'alpha': 1,
        'rho': 0.5,
        'tolerance': 1e-6,
        'print_every': 1
    }
    return config


def minimize(objective, config, start=None):
    '''
    Minimize an objective function starting from a given point.
    
    Args:
        function (callable): The objective function to minimize.
        config (dict): Hyperparameters for the optimization.
        start (list): Starting point for the optimization.
    
    Output:
        x_k (list): The optimal point found by the optimization.
    '''
        
    # extract config parameters
    method = config['method']
    c_decrease = config['c_decrease']
    c_increase = config['c_increase']
    k_max = config['k_max']
    alpha_init = config['alpha']
    rho = config['rho']
    tolerance = config['tolerance']
    print_every = config['print_every']
    
    # get initial guess
    if start is None:
        start = function.get_start()
    x_k = start
    
    
    # print output header
    if len(x_k) > 5:
        print(f'{method} optimization with backtracking linesearch from start = {x_k[:5]}...')
    else:
        print(f'{method} optimization with backtracking linesearch from start = {x_k}')
    print(f'iter                f       ||p_k||       alpha     #func     ||grad_f||')
    
    # evaluate function at start for later comparison
    f_k = objective(x_k)
    H_k = np.eye(len(x_k))
    
    # initialize iteration counter
    k = 1
    
    # optimization loop
    while k < k_max:
        
        # reset hyperparameters for line iteration
        num_calls = 0
        alpha = alpha_init

        # compute gradient at x_k, cuts across all methods       
        grad_k = objective.gradient(x_k)
        
        # compute search direction
        if method == 'SD':
            p_k = -grad_k
        elif method == 'Newton':
            H_k = objective.hessian(x_k)
            p_k = -linalg.solve(H_k, grad_k)
        elif method == 'BFGS':
            p_k = -H_k @ grad_k
        else:
            raise Exception(f'Unknown optimization method: {method}')
        
        # check if search direction is a descent direction
        if grad_k @ p_k > 0:
            print(f'Error: Search direction is not a descent direction')
            return x_k
        
        # perform line search to satisfy Wolfe conditions
        f_k1 = objective(x_k + alpha*p_k)
        grad_k1 = objective.gradient(x_k + alpha*p_k)
        num_calls += 1
        assert c_increase > c_decrease
        upper_boundary = f_k + c_decrease*alpha*(grad_k @ p_k)
        lower_boundary = c_increase*(grad_k @ p_k)
        while f_k1 > upper_boundary or (grad_k1 @ p_k) < lower_boundary:
            alpha *= rho
            f_k1 = objective(x_k + alpha*p_k)
            grad_k1 = objective.gradient(x_k + alpha*p_k)
            num_calls += 1
            if num_calls > 1000:
                print(f'Error: Line search failed to converge')
                if f_k1 > upper_boundary:
                    print(f'Error: f_k1 > upper_boundary')
                if (grad_k @ p_k) < lower_boundary:
                    print(f'Error: dot(grad_k1, p_k) < lower_boundary')
                return x_k
        
        # make the step
        x_k1 = x_k + alpha*p_k
    
        # implement BFGS update 
        if method == 'BFGS':
            s_k = x_k1 - x_k
            y_k = grad_k1 - grad_k
            rho_k = 1/(y_k @ s_k)
            H_k = (np.eye(len(x_k)) - rho_k*np.outer(s_k, y_k)) @ H_k @\
                (np.eye(len(x_k)) - rho_k*np.outer(y_k, s_k)) + rho_k*np.outer(s_k, s_k)
        
        # print iteration output
        max_grad_k = linalg.norm(grad_k, ord=np.inf)
        if k % print_every == 0:
            norm_p_k = linalg.norm(p_k)
            print(f'{k:<5d}      {f_k:<5.4e}      {norm_p_k:<5.2e}    {alpha:<5.2e}         {num_calls:<5d}   {max_grad_k:<5.2e}')
        
        # update variables
        x_k = x_k1
        f_k = f_k1
        k += 1
        
        # check stopping criterion
        if max_grad_k < tolerance:
            print(f'CONVERGED!')
            break
    
    return x_k

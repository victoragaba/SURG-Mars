#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Name: Victor Agaba

Date: 2023-20-11

A module with 3 functions for implementing Newton method and gradient descent:
    init_options()
    minimize()
    factorize_with_mod()
'''

# Import numpy for linear algebra computations
import numpy as np

# Import scipy to use in Cholesky factorization
import scipy as sp


def init_options():
    '''
    Option initialization for the gradient descent algorithm

    Initialize algorithm options with default values

    Return values:
        options:
            This is a dictionary with fields that correspond to algorithmic
            options of our method.  In particular:

            max_iter:
                Maximum number of iterations
            tol:
                Convergence tolerance
            step_type:
                Different ways to calculate the search direction:
                    'gradient_descent'
                    'Newton'
            linesearch:
                Type of line search:
                    'constant'
                    'backtracking'
            alpha_init:
                First trial step size to attempt during line search:
                    'constant':     Value of the constant step size
                    'backtracking': Initial trial step size
            suff_decrease_factor:
                Coefficient in sufficient decrease condition for backtracking
                    line search
            perturb_init:
                Initial perturbation of Hessian matrix to ensure positive
                    definiteness
            perturb_inc_factor:
                Increase factor for Hessian perturbation
            output_level:
                Amount of output to be printed
                0: No output
                1: Only summary information
                2: One line per iteration (good for debugging)
                3: Print iterates in each iteration
    '''
    options = {}
    options['max_iter'] = 1000
    options['tol'] = 1e-6
    options['step_type'] = 'gradient_descent'
    options['linesearch'] = 'backtracking'
    options['alpha_init'] = 1
    options['suff_decrease_factor'] = 1e-4
    options['perturb_init'] = 1e-4
    options['perturb_inc_factor'] = 10
    options['output_level'] = 1

    return options


def minimize(opt_prob, options, x_start=None):
    '''
    Optimization methods for unconstrainted optimization

    This is an implementation of the gradient descent method for unconstrained
    optimization.

    Input arguments:
        opt_prob:
            Optimization problem object to define an unconstrained optimization
            problem. It must have the following methods:

            val = value(x)
                returns value of objective function at x
            grad = gradient(x)
                returns gradient of objective function at x
            x_start = starting_point()
                returns a default starting point
        options:
            This is a structure with options for the algorithm.
            For details see the init_options function.
        x_start:
            Starting point. If set to None, obtain default starting point
            from opt_problem.

    Return values:
        status:
            Return code indicating reason for termination:
            'success': Critical point found (convergence tolerance satisfied)
            'iteration limit':  Maximum number of iterations exceeded
            'error': something went wrong
        x_sol:
            Approximate critical point (or last iterate if there is a failure)
        f_sol:
            function value of x_sol
    '''

    # Get option values
    max_iter = options['max_iter']
    tol = options['tol']
    step_type = options['step_type']
    linesearch = options['linesearch']
    alpha_init = options['alpha_init']
    suff_decrease_factor = options['suff_decrease_factor']
    perturb_init = options['perturb_init']
    perturb_inc_factor = options['perturb_inc_factor']
    output_level = options['output_level']

    # return flag
    # set to error so that status has to be set explicitly in method
    status = 'error'

    # get starting point. If none is provided explicitly for this call,
    # ask the OptProblem object for it.
    if x_start is None:
        x_start = opt_prob.starting_point()
    x_k = np.copy(x_start)

    # Initialize algorithm quantities
    # iteration counter
    iter_count = 0

    # current function value (for output), gradient, and gradient norm
    f_k = opt_prob.value(x_k)
    grad_k = opt_prob.gradient(x_k)
    norm_grad = np.linalg.norm(grad_k, np.inf)
    direc_norm = 0.
    alpha_k = 0.
    num_func_evals = 0
    perturb = 0.
    total_func_evals = 0

    # This flag can be set to zero to terminate the iteration loop early
    keep_going = True

    # Print header and zero-th iteration for output
    if output_level >= 1:
        print(f'Running {step_type} method with {linesearch}', end=' ')
        print(f'line search at step size {alpha_init}\n')
    if output_level >= 2:
        # Print header for output
        # (This is just a fancy way to create a string.  The '%s' formating
        # makes it easy to align the header with the actual output.)
        output_header = '%6s %23s %9s %9s %7s %9s %9s' % \
            ('iter', 'f', '||d_k||', 'alpha', '# func', 'perturb', '||grad||')
        print(output_header)
        print('%6i %23.16e %9.2e %9.2e %7d %9.2e %9.2e' %
              (iter_count, f_k, direc_norm, alpha_k, num_func_evals, perturb,
               norm_grad))
    if output_level >= 3:
        print('Current iterate:', x_k)

    ###########################
    # Main Loop
    ###########################
    while keep_going:

        # Check termination
        if iter_count >= max_iter:
            # Set flag to indicate the maximum number of iterations has been
            # exceeded
            status = 'iteration limit'
            break

        # Reset number of function evaluations
        num_func_evals = 0

        # Compute current "Hessian" using generic Newton algorithm
        if step_type == 'Newton':
            H_k = opt_prob.hessian(x_k)
        elif step_type == 'gradient_descent':
            H_k = np.eye(len(x_k))
        else:
            raise Exception(f'{step_type} is not a valid step_type')

        # Compute Cholesky factor of modified Hessian
        L, perturb = factorize_with_mod(H_k, perturb_init, perturb_inc_factor)

        # Compute search direction if gradient is finite
        if np.isfinite(norm_grad):
            v = sp.linalg.solve(L, -grad_k)
            direc_k = sp.linalg.solve(L.T, v)
        else:
            status = 'divergence'
            break

        # Reset trial step size
        alpha_k = alpha_init

        # Perform line search
        if linesearch == 'backtracking':

            # Backtrack until sufficient decrease condition is satisfied
            f_try = opt_prob.value(x_k + alpha_k*direc_k)
            num_func_evals += 1
            relaxed_tangent = f_k + \
                alpha_k*suff_decrease_factor*np.dot(grad_k, direc_k)
            while f_try > relaxed_tangent:
                # Reduce step size
                alpha_k /= 2

                # Update function value and relaxed tangent
                f_try = opt_prob.value(x_k + alpha_k*direc_k)
                num_func_evals += 1
                relaxed_tangent = f_k + \
                    suff_decrease_factor*alpha_k*np.dot(grad_k, direc_k)

            # Update function value
            f_k = f_try

        # Compute Newton step
        x_k = x_k + alpha_k*direc_k

        # Update function value and gradient
        if linesearch != 'backtracking':
            f_k = opt_prob.value(x_k)
            num_func_evals += 1
        grad_k = opt_prob.gradient(x_k)
        norm_grad = np.linalg.norm(grad_k, np.inf)
        direc_norm = np.linalg.norm(direc_k, np.inf)

        # Update total number of function evaluations
        total_func_evals += num_func_evals

        # Increment iteration counter
        iter_count += 1

        # Iteration output
        if output_level >= 2:

            # Print the output header every 10 iterations
            if iter_count % 10 == 0:
                print(output_header)

            print('%6i %23.16e %9.2e %9.2e %7d %9.2e %9.2e' %
                  (iter_count, f_k, direc_norm, alpha_k, num_func_evals,
                   perturb, norm_grad))

        if output_level >= 3:
            print('Current iterate:', x_k)

        # Check for convergence
        if norm_grad < tol:
            status = 'success'
            keep_going = False

    # Finalize results
    x_sol = np.copy(x_k)
    f_sol = f_k

    # Final output message
    if options['output_level'] >= 1:
        print('')
        print('# Iterations v func evals........:', end=' ')
        print(f'{iter_count} v {total_func_evals}')
        print(f'Final objective..................: {f_sol}')
        print(f'||grad|| at final point..........: {norm_grad}')
        if status == 'success':
            print('Status: Critical point found.')
        elif status == 'iteration limit':
            print('Status: Maximum number of iterations', end=' ')
            print(f'({iter_count}) exceeded.')
        elif status == 'divergence':
            print('Status: Iterates diverged.')
        else:
            raise Exception(f'status has unexpected value: {status}')

    # Return output arguments
    return status, x_sol, f_sol


def factorize_with_mod(H, perturb_init, perturb_inc_factor):
    '''
    Compute Cholesky factor of H + lambda*I where lambda is chosen to make the
    resultant matrix positive definite.

    Input arguments:
        H:
            A symmetric matrix
        perturb_init:
            Initial perturbation to use if H is not positive definite
        perturb_inc_factor:
            Factor by which to increase the perturbation

    Return values:
        L:
            A lower triangular matrix
    '''

    # Initialize matrix to try
    H_try = H

    # Try factorization without perturbation
    try:
        L = sp.linalg.cholesky(H_try, lower=True)

        # If successful, return the factorization
        return L, 0.

    # Otherwise, adjust H_try until it is positive definite
    except np.linalg.LinAlgError:

        # Initialize perturbation
        perturb = perturb_init

        # Compute Cholesky factor
        while True:
            try:
                L = sp.linalg.cholesky(H + perturb*np.eye(len(H)), lower=True)

                # If successful, return the factorization
                return L, perturb

            # Otherwise, increase the perturbation
            except np.linalg.LinAlgError:
                perturb *= perturb_inc_factor

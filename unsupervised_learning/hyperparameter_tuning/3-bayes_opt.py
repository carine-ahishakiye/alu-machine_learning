#!/usr/bin/env python3
"""
Bayesian Optimization on a noiseless 1D Gaussian Process
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Parameters:
        - f: black-box function
        - X_init: np.ndarray (t, 1), initial sampled inputs
        - Y_init: np.ndarray (t, 1), initial sampled outputs
        - bounds: tuple (min, max), search space
        - ac_samples: number of acquisition sample points
        - l: length parameter of kernel
        - sigma_f: standard deviation of output of black-box function
        - xsi: exploration-exploitation factor
        - minimize: bool, whether to minimize or maximize
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        # Generate evenly spaced acquisition sample points
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize


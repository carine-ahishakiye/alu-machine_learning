#!/usr/bin/env python3
"""
Bayesian Optimization - Acquisition function
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement (EI)

        Returns:
        - X_next: np.ndarray of shape (1,), the next best sample point
        - EI: np.ndarray of shape (ac_samples,), expected improvement values
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            # Best observed value so far (minimum)
            mu_sample_opt = np.min(self.gp.Y)
            # Improvement
            imp = mu_sample_opt - mu - self.xsi
        else:
            # Best observed value so far (maximum)
            mu_sample_opt = np.max(self.gp.Y)
            # Improvement
            imp = mu - mu_sample_opt - self.xsi

        # Avoid division by zero
        sigma = sigma.reshape(-1)
        Z = np.zeros_like(mu)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]

        # Expected Improvement
        EI = np.zeros_like(mu)
        EI[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])

        # Next best sample is where EI is maximum
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
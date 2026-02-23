#!/usr/bin/env python3
"""
Bayesian Optimization - Full optimization loop
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
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        sigma = sigma.reshape(-1)
        Z = np.zeros_like(mu)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]

        EI = np.zeros_like(mu)
        EI[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Performs Bayesian optimization

        Parameters:
        - iterations: max number of iterations

        Returns:
        - X_opt: np.ndarray of shape (1,), optimal input
        - Y_opt: np.ndarray of shape (1,), optimal output
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Stop early if X_next already sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            # Evaluate the function at X_next
            Y_next = self.f(X_next)

            # Update the Gaussian process
            self.gp.update(X_next, np.array([Y_next]))

        # Optimal point & value
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt

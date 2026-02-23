#!/usr/bin/env python3
"""
Gaussian Process for 1D noiseless black-box function
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # Initial covariance kernel matrix
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix using the RBF kernel
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - \
                 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process
        """
        K_s = self.kernel(self.X, X_s)        # (t, s)
        K_ss = self.kernel(X_s, X_s)          # (s, s)
        K_inv = np.linalg.inv(self.K)         # (t, t)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with new sample point

        Parameters:
        - X_new: np.ndarray of shape (1,), new sample input
        - Y_new: np.ndarray of shape (1,), new sample output
        """
        # Reshape to (1, 1) for consistency
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)

        # Append new data
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        # Recompute kernel with updated data
        self.K = self.kernel(self.X, self.X)

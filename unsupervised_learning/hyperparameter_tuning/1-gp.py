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

        Parameters:
        - X_s: np.ndarray of shape (s, 1), points to predict

        Returns:
        - mu: np.ndarray of shape (s,), mean for each point in X_s
        - sigma: np.ndarray of shape (s,), variance for each point in X_s
        """
        # Compute covariance matrices
        K_s = self.kernel(self.X, X_s)        # (t, s)
        K_ss = self.kernel(X_s, X_s)          # (s, s)
        K_inv = np.linalg.inv(self.K)         # (t, t)

        # Predictive mean
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Predictive variance
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)

        return mu, sigma

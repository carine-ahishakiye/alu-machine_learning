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

        Parameters:
        - X_init: np.ndarray of shape (t, 1), inputs already sampled
        - Y_init: np.ndarray of shape (t, 1), outputs of black-box for X_init
        - l: length parameter for kernel
        - sigma_f: standard deviation of output of black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # Compute the covariance kernel matrix
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the RBF (Radial Basis Function) kernel.

        Parameters:
        - X1: np.ndarray of shape (m, 1)
        - X2: np.ndarray of shape (n, 1)

        Returns:
        - Covariance kernel matrix as np.ndarray of shape (m, n)
        """
        # Compute squared Euclidean distance
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - \
                 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

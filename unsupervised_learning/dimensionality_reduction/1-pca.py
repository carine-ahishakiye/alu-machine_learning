#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        ndim: new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed X
    """
    X_mean = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_mean)
    W = Vt[:ndim].T
    T = X_mean @ W

    return T
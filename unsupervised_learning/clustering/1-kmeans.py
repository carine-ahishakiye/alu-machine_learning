#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations

    Returns:
        C, clss or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)

    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dists, axis=1)

        C_new = np.copy(C)
        for i in range(k):
            points = X[clss == i]
            if len(points) == 0:
                C_new[i] = np.random.uniform(low, high, size=(d,))
            else:
                C_new[i] = points.mean(axis=0)

        if np.allclose(C, C_new):
            return C_new, clss

        C = C_new

    dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(dists, axis=1)

    return C, clss
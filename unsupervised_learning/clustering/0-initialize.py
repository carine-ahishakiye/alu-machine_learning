#!/usr/bin/env python3
"""K-means clustering initialization"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters

    Returns:
        numpy.ndarray of shape (k, d) with initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if k > n:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, d))

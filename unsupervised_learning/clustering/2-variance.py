#!/usr/bin/env python3
"""Intra-cluster variance"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        C: numpy.ndarray of shape (k, d) containing centroid means

    Returns:
        var or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_dists = np.min(dists, axis=1)
    var = np.sum(min_dists ** 2)

    return var
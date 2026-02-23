#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        var: fraction of variance to maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) containing the weights matrix
    """
    _, s, Vt = np.linalg.svd(X)
    cumvar = np.cumsum(s) / np.sum(s)
    nd = np.argmax(cumvar >= var) + 1
    W = Vt[:nd].T

    return W
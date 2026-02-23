#!/usr/bin/env python3
"""Regular Markov Chain"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain.

    Args:
        P: numpy.ndarray of shape (n, n) representing the transition matrix

    Returns:
        numpy.ndarray of shape (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]

    if np.any(np.linalg.matrix_power(P, 100) <= 0):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady = np.real(eigenvectors[:, idx])
    steady = steady / np.sum(steady)

    return steady.reshape(1, n)
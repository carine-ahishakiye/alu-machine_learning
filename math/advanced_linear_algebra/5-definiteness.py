#!/usr/bin/env python3
"""
Module to compute the definiteness of a square matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a matrix using its eigenvalues.

    Parameters:
    - matrix: a numpy.ndarray of shape (n, n)

    Returns:
    - A string: 'Positive definite', 'Positive semi-definite',
                'Negative definite', 'Negative semi-definite',
                'Indefinite', or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except Exception:
        return None

    if np.all(eigvals > 0):
        return "Positive definite"
    if np.all(eigvals >= 0):
        return "Positive semi-definite"
    if np.all(eigvals < 0):
        return "Negative definite"
    if np.all(eigvals <= 0):
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None


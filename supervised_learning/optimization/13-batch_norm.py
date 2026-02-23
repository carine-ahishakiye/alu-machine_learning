#!/usr/bin/env python3
"""
Batch normalization function
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural network using batch normalization

    Args:
        Z (np.ndarray): shape (m, n) - pre-activation output
        gamma (np.ndarray): shape (1, n) - scale parameter
        beta (np.ndarray): shape (1, n) - offset parameter
        epsilon (float): small number to avoid division by zero

    Returns:
        np.ndarray: normalized Z
    """
    # Compute mean and variance along the batch (axis=0)
    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    # Normalize
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)

    # Scale and shift
    Z_norm = gamma * Z_norm + beta

    return Z_norm

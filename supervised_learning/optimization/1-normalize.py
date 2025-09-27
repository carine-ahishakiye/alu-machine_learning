#!/usr/bin/env python3
"""
Function to normalize (standardize) a dataset using provided mean and std
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    
    Parameters:
    X (np.ndarray): shape (d, nx), dataset to normalize
    m (np.ndarray): shape (nx,), mean of each feature
    s (np.ndarray): shape (nx,), standard deviation of each feature

    Returns:
    X_norm (np.ndarray): normalized dataset
    """
    X_norm = (X - m) / s
    return X_norm

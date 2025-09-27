#!/usr/bin/env python3
"""
Function to calculate normalization constants (mean and std) for a dataset
"""

import numpy as np

def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix
    
    Parameters:
    X (np.ndarray): shape (m, nx), dataset to normalize
        m is the number of data points
        nx is the number of features

    Returns:
    mean, std (np.ndarray, np.ndarray):
        mean: mean of each feature
        std: standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std

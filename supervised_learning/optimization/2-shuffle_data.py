#!/usr/bin/env python3
"""
Function to shuffle two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    
    Parameters:
    X (np.ndarray): shape (m, nx), first dataset
    Y (np.ndarray): shape (m, ny), second dataset
    
    Returns:
    X_shuffled, Y_shuffled (np.ndarray, np.ndarray): shuffled datasets
    """
    # Generate a permutation of indices
    perm = np.random.permutation(X.shape[0])
    
    # Apply permutation to both X and Y
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    
    return X_shuffled, Y_shuffled

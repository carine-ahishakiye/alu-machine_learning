#!/usr/bin/env python3
"""
Policy Gradient functions
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using softmax.

    Args:
        matrix: state matrix (1, n_features)
        weight:  weight matrix (n_features, n_actions)

    Returns:
        softmax probabilities over actions (1, n_actions)
    """
    z = matrix.dot(weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

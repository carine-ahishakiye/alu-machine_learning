#!/usr/bin/env python3
"""
Update a variable using RMSProp optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp

    Parameters:
    alpha (float): learning rate
    beta2 (float): RMSProp decay factor
    epsilon (float): small number to avoid division by zero
    var (np.ndarray): variable to be updated
    grad (np.ndarray): gradient of var
    s (np.ndarray): previous second moment of var

    Returns:
    var_updated (np.ndarray): updated variable
    s (np.ndarray): updated second moment
    """
    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * np.square(grad)
    # Update variable
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s

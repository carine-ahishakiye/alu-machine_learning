
#!/usr/bin/env python3
"""
Update variables using gradient descent with momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the momentum optimization algorithm

    Parameters:
    alpha (float): learning rate
    beta1 (float): momentum weight
    var (np.ndarray or scalar): variable to update
    grad (np.ndarray or scalar): gradient of the variable
    v (np.ndarray or scalar): previous first moment

    Returns:
    var (np.ndarray or scalar): updated variable
    v (np.ndarray or scalar): updated momentum
    """
    # Update the moving average of the gradient
    v = beta1 * v + (1 - beta1) * grad
    # Update the variable
    var = var - alpha * v

    return var, v

#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using gradient descent
with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Performs one step of gradient descent with L2 regularization

    Arguments:
    Y -- one-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights -- dictionary of weights and biases of the neural network
    cache -- dictionary of outputs of each layer of the network
    alpha -- learning rate
    lambtha -- L2 regularization parameter
    L -- number of layers of the network

    Updates weights and biases in place.
    """
    m = Y.shape[1]
    # Initialize dZ for last layer (softmax)
    A_prev = cache["A{}".format(L - 1)]
    A_L = cache["A{}".format(L)]
    dZ = A_L - Y  # derivative for softmax + cross-entropy

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(l - 1)]
        W = weights["W{}".format(l)]

        # Gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights["W{}".format(l)] -= alpha * dW
        weights["b{}".format(l)] -= alpha * db

        # Compute dZ for previous layer if not input layer
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # tanh derivative

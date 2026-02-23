#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    Y: one-hot array of shape (classes, m)
    weights: dictionary of weights and biases
    cache: dictionary containing activations and dropout masks
    alpha: learning rate
    keep_prob: probability a node is kept
    L: number of layers
    Updates weights in place
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y  # derivative for softmax + cross-entropy

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]

        # Compute gradients
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # Backprop through previous layer with tanh and dropout
            dA_prev = np.dot(W.T, dZ)
            D_prev = cache['D' + str(l - 1)]
            dA_prev *= D_prev
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - cache['A' + str(l - 1)] ** 2)  # tanh derivative

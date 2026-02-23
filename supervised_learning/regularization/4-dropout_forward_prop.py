#!/usr/bin/env python3
"""Forward propagation with Dropout"""
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout
    X: numpy.ndarray of shape (nx, m), input data
    weights: dictionary of weights and biases
    L: number of layers
    keep_prob: probability a node is kept
    Returns: dictionary containing layer outputs and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        # Linear combination
        Z = np.dot(W, A_prev) + b

        # Activation
        if l == L:
            # Softmax for last layer
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Tanh for hidden layers
            A = np.tanh(Z)
            # Dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D

        cache['A' + str(l)] = A

    return cache

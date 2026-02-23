#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the L2 regularization cost for a neural network

    Arguments:
    cost -- cost of the network without L2 regularization
    lambtha -- regularization parameter
    weights -- dictionary of the weights and biases of the network
    L -- number of layers
    m -- number of data points used

    Returns:
    cost_with_l2 -- cost accounting for L2 regularization
    """
    l2_sum = 0
    for l in range(1, L + 1):
        W = weights["W{}".format(l)]
        l2_sum += np.sum(np.square(W))
    l2_cost = (lambtha / (2 * m)) * l2_sum
    return cost + l2_cost

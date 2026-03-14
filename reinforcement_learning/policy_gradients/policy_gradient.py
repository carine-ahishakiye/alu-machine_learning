#!/usr/bin/env python3
"""
Policy Gradient functions
"""
import numpy as np


def policy(matrix, weight):
    z = matrix.dot(weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    one_hot = np.zeros(probs.shape[1])
    one_hot[action] = 1

    gradient = state.T.dot((one_hot - probs))

    return action, gradient

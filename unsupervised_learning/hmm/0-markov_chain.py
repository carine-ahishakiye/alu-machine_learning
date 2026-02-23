#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain being in a particular
    state after a specified number of iterations.

    Args:
        P: numpy.ndarray of shape (n, n) representing the transition matrix
        s: numpy.ndarray of shape (1, n) representing the starting state
        t: number of iterations

    Returns:
        numpy.ndarray of shape (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    result = np.matmul(s, np.linalg.matrix_power(P, t))

    return result
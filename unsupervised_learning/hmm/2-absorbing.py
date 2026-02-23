#!/usr/bin/env python3
"""Absorbing Markov Chain"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing.

    Args:
        P: numpy.ndarray of shape (n, n) representing the transition matrix

    Returns:
        True if absorbing, False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False

    n = P.shape[0]
    absorbing_states = np.where(np.diag(P) == 1)[0]

    if len(absorbing_states) == 0:
        return False

    if len(absorbing_states) == n:
        return True

    reachable = set(absorbing_states)
    prev_size = 0

    while len(reachable) != prev_size:
        prev_size = len(reachable)
        for s in range(n):
            if s not in reachable:
                if any(P[s, r] > 0 for r in reachable):
                    reachable.add(s)

    return len(reachable) == n
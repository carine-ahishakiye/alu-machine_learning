#!/usr/bin/env python3
"""Viterbi Algorithm for Hidden Markov Model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a HMM.

    Args:
        Observation: numpy.ndarray of shape (T,) with observation indices
        Emission: numpy.ndarray of shape (N, M) with emission probabilities
        Transition: numpy.ndarray of shape (N, N) with transition probabilities
        Initial: numpy.ndarray of shape (N, 1) with initial state probabilities

    Returns:
        path, P or None, None on failure
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        trans_prob = V[:, t - 1, np.newaxis] * Transition
        B[:, t] = np.argmax(trans_prob, axis=0)
        V[:, t] = np.max(trans_prob, axis=0) * Emission[:, Observation[t]]

    path = [np.argmax(V[:, T - 1])]
    for t in range(T - 1, 0, -1):
        path.append(B[path[-1], t])

    path = path[::-1]
    P = np.max(V[:, T - 1])

    return path, P
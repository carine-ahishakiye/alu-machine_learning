#!/usr/bin/env python3
"""Baum-Welch Algorithm for Hidden Markov Model"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model.

    Args:
        Observations: numpy.ndarray of shape (T,) with observation indices
        Transition: numpy.ndarray of shape (M, M) with transition probabilities
        Emission: numpy.ndarray of shape (M, N) with emission probabilities
        Initial: numpy.ndarray of shape (M, 1) with initial probabilities
        iterations: number of EM iterations

    Returns:
        Transition, Emission or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    if Transition.shape != (M, M):
        return None, None
    if Initial.shape != (M, 1):
        return None, None

    for _ in range(iterations):
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = np.dot(F[:, t - 1], Transition) * \
                Emission[:, Observations[t]]

        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.dot(
                Transition, Emission[:, Observations[t + 1]] * B[:, t + 1])

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = np.dot(np.dot(F[:, t], Transition) *
                           Emission[:, Observations[t + 1]], B[:, t + 1])
            for i in range(M):
                xi[i, :, t] = (F[i, t] * Transition[i, :] *
                                Emission[:, Observations[t + 1]] *
                                B[:, t + 1]) / denom

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma, axis=1, keepdims=True)

        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape(M, 1)))
        for k in range(N):
            Emission[:, k] = np.sum(
                gamma[:, Observations == k], axis=1)
        Emission = Emission / np.sum(gamma, axis=1, keepdims=True)

    return Transition, Emission
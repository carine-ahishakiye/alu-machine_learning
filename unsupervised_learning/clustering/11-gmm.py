#!/usr/bin/env python3
"""GMM using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        pi, m, S, clss, bic
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
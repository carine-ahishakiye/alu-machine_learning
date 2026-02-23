#!/usr/bin/env python3
"""Agglomerative clustering using scipy"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    fig = plt.figure()
    scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist
    )
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')

    return clss
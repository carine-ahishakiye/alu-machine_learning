#!/usr/bin/env python3
"""Neuron class with forward propagation and cost calculation"""

import numpy as np


class Neuron:
    """Neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron
        Args:
            nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron
        Args:
            X (numpy.ndarray): shape (nx, m) input data
        Returns:
            __A: activated output
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression
        Args:
            Y (numpy.ndarray): shape (1, m), correct labels
            A (numpy.ndarray): shape (1, m), activated output
        Returns:
            cost (float)
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

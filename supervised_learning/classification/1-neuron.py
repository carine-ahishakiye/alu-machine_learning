#!/usr/bin/env python3
"""Neuron class that defines a single neuron for binary classification"""

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

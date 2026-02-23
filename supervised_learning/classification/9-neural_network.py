#!/usr/bin/env python3
"""NeuralNetwork class with one hidden layer performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network
        Args:
            nx (int): number of input features
            nodes (int): number of nodes in hidden layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)  # weight matrix
        self.__b1 = np.zeros((nodes, 1))        # bias
        self.__A1 = 0                           # activated output

        # Output neuron parameters
        self.__W2 = np.random.randn(1, nodes)   # weight vector
        self.__b2 = 0                           # bias
        self.__A2 = 0                           # activated output

    # Getters only (no setters)
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

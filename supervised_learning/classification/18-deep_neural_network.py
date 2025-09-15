#!/usr/bin/env python3
"""DeepNeuralNetwork performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx
        for i, nodes in enumerate(layers, 1):
            self.__weights["W" + str(i)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b" + str(i)] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network"""
        self.__cache["A0"] = X
        A = X

        # Only one loop over layers
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache["A" + str(l)] = A

        return A, self.__cache

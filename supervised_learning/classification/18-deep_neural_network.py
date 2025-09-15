#!/usr/bin/env python3
"""18-deep_neural_network.py"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers type and not empty
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx
        # Single loop: validate nodes + initialize weights/biases
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(i + 1)] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights["b" + str(i + 1)] = np.zeros((nodes, 1))
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
        A_prev = X

        # Only one loop over layers
        for l in range(1, self.__L + 1):
            Wl = self.__weights["W" + str(l)]
            bl = self.__weights["b" + str(l)]
            Zl = np.dot(Wl, A_prev) + bl
            A_prev = 1 / (1 + np.exp(-Zl))  # Sigmoid activation
            self.__cache["A" + str(l)] = A_prev

        return A_prev, self.__cache

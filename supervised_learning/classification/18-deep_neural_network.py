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

        # Initialize weights using He et al. method, biases to 0
        layers_dims = [nx] + layers
        for l in range(1, self.__L + 1):
            self.__weights["W" + str(l)] = (np.random.randn(
                layers_dims[l], layers_dims[l - 1])
                * np.sqrt(2 / layers_dims[l - 1])
            )
            self.__weights["b" + str(l)] = np.zeros((layers_dims[l], 1))

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
            # Sigmoid activation
            A_prev = 1 / (1 + np.exp(-Zl))
            self.__cache["A" + str(l)] = A_prev

        return A_prev, self.__cache

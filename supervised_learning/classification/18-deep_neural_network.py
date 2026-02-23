#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""
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
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # One loop allowed for weights/biases initialization
        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights['W' + str(l + 1)] = np.random.randn(
                layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.__weights['b' + str(l + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Intermediate values"""
        return self.__cache

    @property
    def weights(self):
        """Weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network"""
        self.__cache['A0'] = X
        # One loop allowed for forward propagation
        for l in range(self.__L):
            W = self.__weights['W' + str(l + 1)]
            b = self.__weights['b' + str(l + 1)]
            A_prev = self.__cache['A' + str(l)]
            Z = np.dot(W, A_prev) + b
            self.__cache['A' + str(l + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

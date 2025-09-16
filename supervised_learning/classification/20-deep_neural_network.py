#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network
        nx: number of input features
        layers: list representing number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}       # dictionary to hold intermediary values
        self.__weights = {}     # dictionary to hold weights and biases

        # He initialization
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            layer_key_W = 'W' + str(i + 1)
            layer_key_b = 'b' + str(i + 1)

            if i == 0:
                self.__weights[layer_key_W] = (
                    np.random.randn(layers[i], nx) *
                    np.sqrt(2.0 / nx)
                )
            else:
                self.__weights[layer_key_W] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2.0 / layers[i - 1])
                )
            self.__weights[layer_key_b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network
        X: input data of shape (nx, m)
        Returns output and cache
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid
            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression
        Y: correct labels (1, m)
        A: predicted output (1, m)
        Returns cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A + 1e-8) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        X: input data (nx, m)
        Y: correct labels (1, m)
        Returns prediction and cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

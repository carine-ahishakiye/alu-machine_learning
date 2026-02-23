#!/usr/bin/env python3
"""
Deep Neural Network performing binary classification
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(self.__L):
            if not isinstance(layers[layer_idx], int) or layers[layer_idx] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if layer_idx == 0:
                he_init = np.sqrt(2 / nx)
                self.__weights['W1'] = np.random.randn(
                    layers[0], nx) * he_init
            else:
                he_init = np.sqrt(2 / layers[layer_idx - 1])
                self.__weights['W' + str(layer_idx + 1)] = (
                    np.random.randn(layers[layer_idx],
                                    layers[layer_idx - 1]) * he_init
                )
            self.__weights['b' + str(layer_idx + 1)] = np.zeros(
                (layers[layer_idx], 1)
            )

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation"""
        self.__cache['A0'] = X

        for layer_idx in range(1, self.__L + 1):
            W = self.__weights['W' + str(layer_idx)]
            b = self.__weights['b' + str(layer_idx)]
            A_prev = self.__cache['A' + str(layer_idx - 1)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(layer_idx)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost using logistic regression"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A + 1e-8) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent
        Y: correct labels (1, m)
        cache: intermediary values
        alpha: learning rate
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = cache['A' + str(self.__L)] - Y  # output layer error

        for layer_idx in reversed(range(1, self.__L + 1)):
            A_prev = cache['A' + str(layer_idx - 1)]
            W = weights_copy['W' + str(layer_idx)]

            # Gradients
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Update weights
            self.__weights['W' + str(layer_idx)] = W - alpha * dW
            self.__weights['b' + str(layer_idx)] = (
                self.__weights['b' + str(layer_idx)] - alpha * db
            )

            if layer_idx > 1:
                A_prev = cache['A' + str(layer_idx - 1)]
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

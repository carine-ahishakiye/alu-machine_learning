#!/usr/bin/env python3
"""28-deep_neural_network.py"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for l in range(1, self.__L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]
            # He initialization
            self.__weights['W' + str(l)] = (np.random.randn(layer_size, prev_size)
                                            * np.sqrt(2 / prev_size))
            self.__weights['b' + str(l)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Calculates forward propagation of the network"""
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]
            Zl = np.dot(Wl, Al_prev) + bl
            if l != self.__L:
                if self.__activation == 'sig':
                    Al = 1 / (1 + np.exp(-Zl))
                else:  # tanh
                    Al = np.tanh(Zl)
            else:
                # Output layer uses sigmoid
                Al = 1 / (1 + np.exp(-Zl))
            self.__cache['A' + str(l)] = Al
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions and cost"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = self.__cache['A' + str(self.__L)] - Y

        for l in reversed(range(1, self.__L + 1)):
            Al_prev = self.__cache['A' + str(l - 1)]
            Wl = weights_copy['W' + str(l)]
            bl = weights_copy['b' + str(l)]

            dW = np.dot(dZ, Al_prev.T) / m

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
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for layer_idx in range(1, self.__L + 1):
            layer_size = layers[layer_idx - 1]
            prev_size = nx if layer_idx == 1 else layers[layer_idx - 2]
            # He initialization
            self.__weights['W' + str(layer_idx)] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights['b' + str(layer_idx)] = np.zeros((layer_size, 1))

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
        for layer_idx in range(1, self.__L + 1):
            W = self.__weights['W' + str(layer_idx)]
            b = self.__weights['b' + str(layer_idx)]
            A_prev = self.__cache['A' + str(layer_idx - 1)]
            Z = np.dot(W, A_prev) + b
            if layer_idx != self.__L:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)
            else:
                A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(layer_idx)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates network predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = self.__cache['A' + str(self.__L)] - Y

        for layer_idx in reversed(range(1, self.__L + 1)):
            A_prev = self.__cache['A' + str(layer_idx - 1)]
            W = weights_copy['W' + str(layer_idx)]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_idx > 1:
                A_prev_Z = self.__cache['A' + str(layer_idx - 1)]
                if self.__activation == 'sig':
                    dZ = np.dot(W.T, dZ) * (A_prev_Z * (1 - A_prev_Z))
                else:
                    dZ = np.dot(W.T, dZ) * (1 - A_prev_Z**2)

            self.__weights['W' + str(layer_idx)] -= alpha * dW
            self.__weights['b' + str(layer_idx)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        costs, steps_list = [], []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps_list.append(i)
            self.gradient_descent(Y, alpha)
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(steps_list, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        import os
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)

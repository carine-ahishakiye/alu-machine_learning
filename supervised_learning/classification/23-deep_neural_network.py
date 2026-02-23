#!/usr/bin/env python3
"""
DeepNeuralNetwork class with training capability
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list with number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}
        self.__weights = {}

        # He et al. initialization
        for i in range(self.__L):
            if i == 0:
                he_init = np.sqrt(2 / nx)
                self.__weights['W1'] = np.random.randn(layers[0], nx) * he_init
                self.__weights['b1'] = np.zeros((layers[0], 1))
            else:
                he_init = np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) * he_init
                )
                self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:  # output layer
                A = 1 / (1 + np.exp(-Z))
            else:  # hidden layers use tanh or relu
                A = np.tanh(Z)
            self.__cache['A' + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the logistic regression cost"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A + 1e-8) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = self.__cache['A' + str(self.__L)] - Y

        for i in reversed(range(1, self.__L + 1)):
            A_prev = self.__cache['A' + str(i - 1)]
            W = weights_copy['W' + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                A_prev = self.__cache['A' + str(i - 1)]
                dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)  # derivative tanh

            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        """
        # Validate iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Validate step only if needed
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration_list = []

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                iteration_list.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            if i < iterations:
                self.gradient_descent(Y, alpha)

        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

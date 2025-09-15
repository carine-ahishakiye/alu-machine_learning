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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output neuron parameters
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getters
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

    def forward_prop(self, X):
        """
        Calculates forward propagation
        Args:
            X (ndarray): shape (nx, m), input data
        Returns:
            A1, A2 (ndarray): activated outputs of hidden and output layers
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y (ndarray): shape (1, m), correct labels
            A (ndarray): shape (1, m), activated outputs
        Returns:
            cost (float): logistic regression cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Args:
            X (ndarray): input data, shape (nx, m)
            Y (ndarray): correct labels, shape (1, m)
        Returns:
            prediction (ndarray): predicted labels, shape (1, m)
            cost (float): cost of the model
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)

        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent
        Args:
            X (ndarray): input data of shape (nx, m)
            Y (ndarray): true labels of shape (1, m)
            A1 (ndarray): activated output of hidden layer
            A2 (ndarray): predicted output of output neuron
            alpha (float): learning rate
        Updates:
            __W1, __b1, __W2, __b2
        """
        m = X.shape[1]

        # Output layer gradients
        dZ2 = A2 - Y                         # (1, m)
        dW2 = (1 / m) * np.matmul(dZ2, A1.T) # (1, nodes)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer gradients
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))  # (nodes, m)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)                  # (nodes, nx)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)   # (nodes, 1)

        # Update parameters
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

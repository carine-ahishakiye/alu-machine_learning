#!/usr/bin/env python3
"""
Neuron class with forward propagation, cost, evaluation,
gradient descent, and advanced training method with verbose
and graph options.
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron
        Args:
            nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """Calculate forward propagation"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Compute logistic regression cost"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluate neuron predictions and cost"""
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent"""
        m = Y.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neuron
        Args:
            X (numpy.ndarray): input data, shape (nx, m)
            Y (numpy.ndarray): correct labels, shape (1, m)
            iterations (int): number of iterations
            alpha (float): learning rate
            verbose (bool): print cost every step
            graph (bool): plot cost graph
            step (int): interval for verbose/graph
        Returns:
            tuple: evaluation of training data (predictions, cost)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            if graph and (i % step == 0 or i == 0 or i == iterations):
                costs.append(cost)
                steps.append(i)

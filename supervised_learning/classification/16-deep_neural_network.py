#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network for binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network
        nx: number of input features
        layers: list representing the number of nodes in each layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)        # number of layers
        self.cache = {}             # store intermediary values
        self.weights = {}           # store weights and biases

        # Single loop to initialize weights and biases
        for i in range(self.L):
            # layer sizes
            layer_size = layers[i]
            prev_size = nx if i == 0 else layers[i - 1]

            # He initialization for weights
            self.weights['W' + str(i + 1)] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(2 / prev_size)
            )

            # Zero initialization for biases
            self.weights['b' + str(i + 1)] = np.zeros((layer_size, 1))

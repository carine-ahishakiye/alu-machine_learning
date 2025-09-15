#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor

        Args:
            nx (int): number of input features
            layers (list): list representing the number of nodes in each layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)      # number of layers
        self.cache = {}           # dictionary to hold intermediary values
        self.weights = {}         # dictionary to hold weights and biases

        # Initialize weights and biases using one loop
        for l in range(self.L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]

            # He initialization for weights
            self.weights['W' + str(l + 1)] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(2 / prev_size)
            )

            # Bias initialized to zeros
            self.weights['b' + str(l + 1)] = np.zeros((layer_size, 1))

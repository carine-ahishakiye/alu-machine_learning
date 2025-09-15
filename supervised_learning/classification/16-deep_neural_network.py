#!/usr/bin/env python3
"""Deep Neural Network performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

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

        self.L = len(layers)       # number of layers
        self.cache = {}            # store intermediate values
        self.weights = {}          # store weights and biases

        # Initialize weights and biases using He et al. method
        for l in range(1, self.L + 1):
            if l == 1:
                prev_nodes = nx
            else:
                prev_nodes = layers[l - 2]

            self.weights["W" + str(l)] = (
                np.random.randn(layers[l - 1], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )
            self.weights["b" + str(l)] = np.zeros((layers[l - 1], 1))

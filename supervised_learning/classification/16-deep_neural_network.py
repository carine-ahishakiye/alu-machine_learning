#!/usr/bin/env python3
"""16-deep_neural_network.py"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # Number of layers
        self.L = len(layers)
        # Cache for intermediate values
        self.cache = {}
        # Weights and biases
        self.weights = {}

        # Initialize weights and biases with one loop
        for l in range(1, self.L + 1):
            nodes = layers[l - 1]
            prev_nodes = nx if l == 1 else layers[l - 2]
            self.weights["W" + str(l)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.weights["b" + str(l)] = np.zeros((nodes, 1))

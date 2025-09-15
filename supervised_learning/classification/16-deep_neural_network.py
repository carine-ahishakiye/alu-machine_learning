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

        # Initialize attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev_nodes = nx
        # Only one loop allowed: validate layers and initialize weights/biases
        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W" + str(i + 1)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.weights["b" + str(i + 1)] = np.zeros((nodes, 1))
            prev_nodes = nodes

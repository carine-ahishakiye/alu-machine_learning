#!/usr/bin/env python3
"""
Module defines a DeepNeuralNetwork performing binary classification
and supporting save/load using pickle.
"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification."""

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list containing number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or \
           not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights with He et al. method
        for i in range(self.L):
            layer_size = layers[i]
            prev_size = nx if i == 0 else layers[i - 1]
            self.weights['W' + str(i + 1)] = \
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.weights['b' + str(i + 1)] = np.zeros((layer_size, 1))

    # Implement forward_prop, cost, evaluate, gradient_descent, train methods here
    # (use previous 23-deep_neural_network.py code)

    def save(self, filename):
        """Save the instance object to a pickle file."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)

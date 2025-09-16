#!/usr/bin/env python3
"""
DeepNeuralNetwork with save/load methods
"""
import numpy as np
import pickle
import os

class DeepNeuralNetwork:
    # existing __init__, forward_prop, cost, evaluate, gradient_descent, train methods here

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        Args:
            filename: name of file to save object to
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        Args:
            filename: file to load object from
        Returns:
            loaded object or None if file doesn't exist
        """
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)

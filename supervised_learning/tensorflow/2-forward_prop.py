#!/usr/bin/env python3
"""
This module contains a function to create the forward propagation
graph for a neural network using the create_layer function.
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for input data
        layer_sizes (list): number of nodes in each layer
        activations (list): activation functions for each layer

    Returns:
        Tensor output of the final layer (network prediction)
    """
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A

#!/usr/bin/env python3
"""
This module contains a function to create the training operation for a neural network.
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Args:
        loss: tensor containing the loss of the network’s prediction
        alpha: learning rate

    Returns:
        TensorFlow operation that trains the network
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op

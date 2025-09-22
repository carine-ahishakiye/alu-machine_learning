#!/usr/bin/env python3
"""
This module contains a function to create the training operation
for a neural network using gradient descent.
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates a training operation that minimizes the loss
    using gradient descent.

    Args:
        loss: tensor containing the loss of the network
        alpha: learning rate

    Returns:
        TensorFlow operation to train the network
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op

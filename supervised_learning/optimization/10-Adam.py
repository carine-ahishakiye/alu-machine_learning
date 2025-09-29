#!/usr/bin/env python3
"""
Adam optimization operation creator
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm

    Args:
        loss: tensor containing the loss of the network
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero

    Returns:
        The Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    train_op = optimizer.minimize(loss)
    return train_op

#!/usr/bin/env python3
"""
Batch normalization layer creation for TensorFlow 1.x
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network

    Args:
        prev: tf.Tensor, activated output of the previous layer
        n: int, number of nodes in the layer
        activation: activation function to apply

    Returns:
        tf.Tensor: activated output of the batch-normalized layer
    """
    # Dense layer with variance scaling initializer
    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
        use_bias=False  # bias will be replaced by beta in batch norm
    )(prev)

    # Batch norm parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    epsilon = 1e-8

    # Compute batch mean and variance
    batch_mean, batch_var = tf.nn.moments(dense, axes=[0])

    # Normalize
    Z_norm = tf.nn.batch_normalization(dense, batch_mean, batch_var, beta, gamma, epsilon)

    # Apply activation
    A = activation(Z_norm)

    return A

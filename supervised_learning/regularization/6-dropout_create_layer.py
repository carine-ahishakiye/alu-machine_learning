#!/usr/bin/env python3
"""Creates a TensorFlow layer with Dropout"""
import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev: tensor from previous layer
    n: number of nodes in the layer
    activation: activation function to use
    keep_prob: probability that a node will be kept
    Returns: output tensor of the layer with dropout applied
    """
    # Weight initialization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    W = tf.get_variable("W", shape=(prev.get_shape()[1], n), initializer=initializer)
    b = tf.get_variable("b", shape=(n,), initializer=tf.zeros_initializer())

    # Linear computation
    Z = tf.add(tf.matmul(prev, W), b)

    # Apply activation function if specified
    A = activation(Z) if activation is not None else Z

    # Apply dropout during training
    A_dropout = tf.nn.dropout(A, rate=1 - keep_prob)  # TensorFlow 1.x uses 'rate=1-keep_prob'

    return A_dropout

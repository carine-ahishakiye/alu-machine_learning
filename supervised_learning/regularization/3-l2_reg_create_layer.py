#!/usr/bin/env python3
"""Creates a TensorFlow layer with L2 regularization"""
import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev: tensor output from previous layer
    n: number of nodes in the new layer
    activation: activation function for the layer
    lambtha: L2 regularization parameter

    Returns: tensor output of the new layer
    """
    # Initialize weights with L2 regularization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_reg = tf.keras.regularizers.l2(lambtha)
    weights = tf.get_variable(
        "weights", shape=(prev.get_shape()[1], n),
        initializer=initializer, regularizer=l2_reg
    )
    
    # Initialize biases
    biases = tf.get_variable("biases", shape=(n,), initializer=tf.zeros_initializer())
    
    # Linear combination
    z = tf.add(tf.matmul(prev, weights), biases)
    
    # Apply activation function if provided
    return activation(z) if activation is not None else z

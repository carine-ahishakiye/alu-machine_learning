#!/usr/bin/env python3
"""
This module contains a function to create a fully connected layer
with He initialization for the weights.
"""

import tensorflow as tf


def create_layer(prev, n, activation):

    he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=he_init,
        name="layer"
    )(prev)
    return layer

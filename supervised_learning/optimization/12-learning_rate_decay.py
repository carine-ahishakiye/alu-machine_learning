#!/usr/bin/env python3
"""
Learning rate decay operation using inverse time decay in TensorFlow
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates the learning rate decay operation in tensorflow
    using inverse time decay (stepwise)

    Args:
        alpha (float): original learning rate
        decay_rate (float): rate at which alpha decays
        global_step (int): number of passes of gradient descent elapsed
        decay_step (int): number of passes before further decay

    Returns:
        tf.Tensor: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True  # ensures stepwise decay
    )

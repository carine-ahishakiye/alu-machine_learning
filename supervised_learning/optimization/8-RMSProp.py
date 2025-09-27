#!/usr/bin/env python3
"""
Creates the RMSProp training operation for TensorFlow
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp training operation in TensorFlow

    Parameters:
    loss (tf.Tensor): the loss of the network
    alpha (float): learning rate
    beta2 (float): RMSProp decay factor
    epsilon (float): small number to avoid division by zero

    Returns:
    train_op (tf.Operation): the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=0.0,
                                          momentum=0.0,
                                          epsilon=epsilon,
                                          centered=False)
    train_op = optimizer.minimize(loss)
    return train_op

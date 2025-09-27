#!/usr/bin/env python3
"""
Create a momentum optimization operation in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation using gradient descent with momentum

    Parameters:
    loss (tf.Tensor): loss tensor of the network
    alpha (float): learning rate
    beta1 (float): momentum weight

    Returns:
    train_op (tf.Operation): operation to perform one step of training
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op

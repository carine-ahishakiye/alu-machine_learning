#!/usr/bin/env python3
"""
This module contains a function to calculate the softmax cross-entropy loss.
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        Tensor containing the loss of the prediction
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
    return loss

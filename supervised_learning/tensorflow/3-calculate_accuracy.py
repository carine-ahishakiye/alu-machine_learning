#!/usr/bin/env python3
"""
This module contains a function to calculate the accuracy of a prediction.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of predictions.

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        Tensor containing the decimal accuracy of the prediction
    """
    # Compare predicted labels to true labels
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    # Compute the mean of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

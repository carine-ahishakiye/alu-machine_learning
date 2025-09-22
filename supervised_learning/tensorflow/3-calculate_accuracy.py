#!/usr/bin/env python3
"""
This module contains a function to calculate the accuracy of predictions
for a neural network.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of predictions.

    Args:
        y: placeholder for true labels (one-hot)
        y_pred: tensor containing predicted outputs (logits or probabilities)

    Returns:
        Tensor containing the decimal accuracy
    """
    # Get predicted class indices
    y_pred_class = tf.argmax(y_pred, axis=1)
    # Get true class indices
    y_true_class = tf.argmax(y, axis=1)
    # Compare predicted vs true
    correct_predictions = tf.equal(y_pred_class, y_true_class)
    # Convert boolean to float and take mean
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

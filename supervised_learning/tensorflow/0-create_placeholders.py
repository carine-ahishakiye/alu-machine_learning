#!/usr/bin/env python3
"""
This module contains a function to create TensorFlow placeholders
for a neural network classifier.
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for a neural network.

    Args:
        nx (int): Number of input features
        classes (int): Number of classes for classification

    Returns:
        x: placeholder for input data
        y: placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y

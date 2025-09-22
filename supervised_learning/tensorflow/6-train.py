#!/usr/bin/env python3
"""
This module contains a function to build, train, and save a neural network.
"""

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: training input data
        Y_train: training labels
        X_valid: validation input data
        Y_valid: validation labels
        layer_sizes: list of nodes in each layer
        activations: list of activation functions
        alpha: learning rate
        iterations: number of iterations to train
        save_path: path to save the trained model

    Returns:
        The path where the model was saved
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    
    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    
    # Loss, accuracy, and training operation
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    
    # Add to collection
    tf.add_to_c

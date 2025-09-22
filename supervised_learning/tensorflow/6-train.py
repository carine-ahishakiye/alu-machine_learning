#!/usr/bin/env python3
"""
This module contains a function to build, train, and save a neural network classifier.
"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: training input data, numpy.ndarray
        Y_train: training labels, one-hot numpy.ndarray
        X_valid: validation input data
        Y_valid: validation labels, one-hot
        layer_sizes: list of nodes per layer
        activations: list of activation functions per layer
        alpha: learning rate
        iterations: number of iterations to train
        save_path: path to save the model

    Returns:
        Path where the model was saved
    """
    # Create placeholders
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Loss, accuracy, and training operation
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add to collections
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            # Compute training and validation metrics
            train_cost, train_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_valid, y: Y_valid})

            # Print at 0th, every 100 iterations, and last iteration
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

            # Perform a training step (skip at last iteration)
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model
        saver.save(sess, save_path)

    return save_path

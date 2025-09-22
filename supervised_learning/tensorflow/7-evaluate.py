#!/usr/bin/env python3
"""
This module contains a function to evaluate a trained neural network.
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X: numpy.ndarray containing the input data
        Y: numpy.ndarray containing the one-hot labels
        save_path: path to load the trained model from

    Returns:
        Y_pred: network predictions
        accuracy: accuracy of the network on X
        loss: loss of the network on X
    """
    # Start a session
    with tf.Session() as sess:
        # Load the meta graph and restore weights
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve tensors from the collection
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Compute predictions, accuracy, and loss
        Y_pred, acc, cost = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={tf.get_collection('x')[0]: X,
                       tf.get_collection('y')[0]: Y}
        )

    return Y_pred, acc, cost

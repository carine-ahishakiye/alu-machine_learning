#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Adds L2 regularization penalties to the original cost

    Arguments:
    cost -- tensor containing the cost without L2 regularization

    Returns:
    l2_cost -- tensor containing the cost accounting for L2 regularization
    """
    # Collect all L2 regularization losses from the graph
    l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # Sum the original cost with the L2 losses
    l2_cost = cost + tf.reduce_sum(l2_losses)
    
    return l2_cost

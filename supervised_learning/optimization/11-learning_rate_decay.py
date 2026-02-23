#!/usr/bin/env python3
"""
Learning rate decay function using inverse time decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay (stepwise)

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight controlling decay rate
        global_step (int): number of passes of gradient descent elapsed
        decay_step (int): number of passes before further decay

    Returns:
        float: the updated learning rate
    """
    decay_factor = global_step // decay_step
    updated_alpha = alpha / (1 + decay_rate * decay_factor)
    return updated_alpha

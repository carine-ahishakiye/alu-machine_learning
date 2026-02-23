#!/usr/bin/env python3
"""
Calculates the bias-corrected weighted moving average of a dataset
"""


def moving_average(data, beta):
    """
    Computes the moving average of a list of numbers with bias correction

    Parameters:
    data (list): list of numeric data points
    beta (float): weight used for moving average (0 < beta < 1)

    Returns:
    list: bias-corrected moving averages
    """
    v = 0
    moving_averages = []

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta**t)  # bias correction
        moving_averages.append(v_corrected)

    return moving_averages

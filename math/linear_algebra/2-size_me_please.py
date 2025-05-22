#!/usr/bin/env python3
"""
This module contains a function to calculate the shape of a nested list (matrix).
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix (nested lists).

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the size in each dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape

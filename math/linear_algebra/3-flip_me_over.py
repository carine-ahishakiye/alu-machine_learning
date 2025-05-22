#!/usr/bin/env python3
"""
Module for matrix transpose function.
"""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The input 2D matrix.

    Returns:
        list of lists: The transposed matrix.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = []

    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(matrix[r][c])
        transposed.append(new_row)

    return transposed:wq


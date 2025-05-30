#!/usr/bin/env python3
"""
This module contains a function that calculates
the minor matrix of a square matrix.
"""


def determinant(matrix):
    """
    Recursively calculates the determinant of a matrix.

    Args:
        matrix (list of lists): The input square matrix.

    Returns:
        int or float: The determinant of the matrix.
    """
    if matrix == [[]]:
        return 1
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for col in range(size):
        sub = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.

#!/usr/bin/env python3
"""
Module that contains a function to calculate
the determinant of a square matrix without using imports.
"""


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): A list of lists representing the matrix.

    Returns:
        int or float: The determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a square matrix")

    if size == 1:
        return matrix[0][0]

    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(size):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det

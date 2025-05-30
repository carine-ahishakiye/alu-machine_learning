#!/usr/bin/env python3
"""
Calculates the determinant of a matrix.
"""


def determinant(matrix):
    """Calculate the determinant of a matrix."""

    # Validate matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle empty matrix (0x0)
    if matrix == [[]]:
        return 1

    # Get dimensions
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case for n > 2
    det = 0
    for col in range(n):
        # Build minor matrix
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det

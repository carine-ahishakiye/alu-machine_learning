#!/usr/bin/env python3
"""This module defines a function to calculate the minor matrix of a square matrix."""


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): A square matrix.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]


    def determinant(m):
        """Recursively computes the determinant of a square matrix."""
        if len(m) == 1:
            return m[0][0]
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        det = 0
        for col in range(len(m)):
            minor = [row[:col] + row[col + 1:] for row in m[1:]]
            sign = -1 if col % 2 else 1
            det += sign * m[0][col] * determinant(minor)
        return det


    minors = []
    for i in range(len(matrix)):
        row_minors = []
        for j in range(len(matrix)):
            minor_ij = [row[:j] + row[j + 1:] for k, row in enumerate(matrix)
                        if k != i]
            row_minors.append(determinant(minor_ij))
        minors.append(row_minors)

    return minors

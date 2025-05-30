#!/usr/bin/env python3
"""
Function to compute the minor matrix of a given square matrix.
"""

def determinant(matrix):
    """
    Recursively calculates the determinant of a matrix.
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
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)
    return det

def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.
    """
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if matrix == [[matrix[0][0]]]:
        return [[1]]

    size = len(matrix)
    minors = []

    for i in range(size):
        row_minors = []
        for j in range(size):
            sub = [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]
            row_minors.append(determinant(sub))
        minors.append(row_minors)

    return minors

#!/usr/bin/env python3
"""
Calculates the cofactor matrix of a square matrix.
"""


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.
    """
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for j in range(size):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1)**j) * matrix[0][j] * determinant(minor)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if size == 1:
        return [[1]]

    minors = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            submatrix = [row[:j] + row[j+1:] for idx, row in enumerate(matrix) if idx != i]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)
    return minors


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)
    cofactors = []
    for i in range(size):
        row = []
        for j in range(size):
            sign = (-1) ** (i + j)
            row.append(sign * minors[i][j])
        cofactors.append(row)
    return cofactors

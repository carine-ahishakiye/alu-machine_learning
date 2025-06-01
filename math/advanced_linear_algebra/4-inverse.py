#!/usr/bin/env python3
"""
Module to compute the inverse of a square matrix.
"""


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.
    """
    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    result = 0
    for col in range(len(matrix)):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        sign = (-1) ** col
        result += sign * matrix[0][col] * determinant(minor)
    return result


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.
    """
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cof_matrix = []
    for i in range(len(matrix)):
        row_cof = []
        for j in range(len(matrix)):
            minor = [
                r[:j] + r[j + 1:]
                for k, r in enumerate(matrix) if k != i
            ]
            sign = (-1) ** (i + j)
            row_cof.append(sign * determinant(minor))
        cof_matrix.append(row_cof)
    return cof_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix (transpose of cofactor matrix).
    """
    cof = cofactor(matrix)
    return [list(row) for row in zip(*cof)]


def inverse(matrix):
    """
    Calculates the inverse of a square matrix.

    Returns None if the matrix is singular.
    """
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    if len(matrix) == 1:
        return [[1 / matrix[0][0]]]

    adj = adjugate(matrix)
    inv = [[elem / det for elem in row] for row in adj]
    return inv

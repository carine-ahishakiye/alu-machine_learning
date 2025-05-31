#!/usr/bin/env python3
"""
Module for calculating the cofactor matrix of a given square matrix.
"""

def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.
    """
    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    result = 0
    for col in range(len(matrix[0])):
        minor = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]
        sign = (-1) ** col
        result += sign * matrix[0][col] * determinant(minor)
    return result


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a given square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                                for row in matrix):
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
                for k, r in enumerate(matrix)
                if k != i
            ]
            sign = (-1) ** (i + j)
            cofactor_val = sign * determinant(minor)
            row_cof.append(cofactor_val)
        cof_matrix.append(row_cof)
    return cof_matrix

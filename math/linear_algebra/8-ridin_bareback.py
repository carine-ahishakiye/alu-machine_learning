#!/usr/bin/env python3
"""
This module provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices and returns the result.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list:
            A new matrix representing the product of mat1 and mat2,
            or None if the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            product = sum(a * b for a, b in zip(row, col))
            new_row.append(product)
        result.append(new_row)

    return result

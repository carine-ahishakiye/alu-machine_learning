#!/usr/bin/env python3
"""
This module provides a function to concatenate two 2D matrices
along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.
        axis (int): Axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
        list of list:
            A new matrix representing the concatenation,
            or None if the matrices cannot be concatenated.
    """
    if axis == 0:
        # Check if number of columns match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]

    elif axis == 1:
        # Check if number of rows match
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    return None

#!/usr/bin/env python3
"""
Function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.

    Returns:
        list: New 2D matrix with sums of corresponding elements, or None if shapes differ.
    """
    if len(mat1) != len(mat2):
        return None

    # Check if each row has the same length
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # Add matrices element-wise
    return [[elem1 + elem2 for elem1, elem2 in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]

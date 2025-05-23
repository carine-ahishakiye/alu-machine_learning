#!/usr/bin/env python3
"""
Module for element-wise operations on numpy ndarrays.
"""


def np_elementwise(mat1, mat2):
    """Perform element-wise add, sub, mul, and div on mat1 and mat2."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

#!/usr/bin/env python3
"""
This module contains a function `matrix_shape` that calculates the shape of a nested list (matrix).
"""

def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape

#!/usr/bin/env python3
"""
This module provides a function to concatenate two arrays
of integers or floats into a new list.
"""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays and returns a new list.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list: A new list containing all elements of arr1
              followed by all elements of arr2.
    """
    return arr1 + arr2

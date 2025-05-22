#!/usr/bin/env python3
"""
Function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Args:
        arr1 (list of int/float): First array.
        arr2 (list of int/float): Second array.

    Returns:
        list: New list with element-wise sums, or None if lengths differ.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]

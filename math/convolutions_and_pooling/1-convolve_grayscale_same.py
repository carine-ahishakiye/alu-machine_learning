#!/usr/bin/env python3
"""Module that performs a same convolution on grayscale images."""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w) containing multiple grayscale
images
    - kernel (numpy.ndarray): shape (kh, kw) containing the kernel for the
convolution

    Returns:
    - numpy.ndarray: shape (m, h, w) containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

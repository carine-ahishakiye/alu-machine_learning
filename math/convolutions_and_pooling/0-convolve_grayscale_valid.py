#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
      Multiple grayscale images
    - kernel: numpy.ndarray of shape (kh, kw)
      Kernel for the convolution

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return output

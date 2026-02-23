#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images with custom
padding."""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w), multiple grayscale images
    - kernel (numpy.ndarray): shape (kh, kw), the kernel for the convolution
    - padding (tuple): (ph, pw), the padding for the height and width

    Returns:
    - numpy.ndarray: convolved images of shape (m, h + 2*ph - kh + 1, w + 2*pw
- kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

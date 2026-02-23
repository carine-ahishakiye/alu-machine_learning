#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images with custom
padding."""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w), multiple grayscale images
    - kernel (numpy.ndarray): shape (kh, kw), the kernel for the convolution
    - padding (str or tuple): 'same', 'valid', or (ph, pw)
    - stride (tuple): (sh, sw), the stride for the convolution

    Returns:
    - numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:  # 'valid'
        ph = pw = 0

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    out_h = (padded.shape[1] - kh) // sh + 1
    out_w = (padded.shape[2] - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

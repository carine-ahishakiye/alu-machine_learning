#!/usr/bin/env python3
"""Module that performs a convolution on images with padding and stride."""


import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale or RGB images.

    Parameters:
    - images: np.ndarray of shape (m, h, w) or (m, h, w, c)
    - kernel: np.ndarray of shape (kh, kw) or (kh, kw, c)
    - padding: 'same', 'valid', or (ph, pw)
    - stride: tuple of (sh, sw)

    Returns:
    - np.ndarray: convolved images
    """
    if images.ndim == 3:
        m, h, w = images.shape
        c = 1
        images = images[..., np.newaxis]
        kernel = kernel[..., np.newaxis]
    elif images.ndim == 4:
        m, h, w, c = images.shape
    else:
        raise ValueError("Images must be 3D or 4D")

    kh, kw = kernel.shape[:2]
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("Invalid padding type")

    images_padded = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = images_padded[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output

#!/usr/bin/env python3
"""Module that performs a convolution on RGB images (multi-channel)."""


import numpy as np


def convolve_channels(images, kernel, padding='valid', stride=(1, 1)):
    """
    Performs a convolution on RGB images.

    Parameters:
    - images: np.ndarray of shape (m, h, w, c)
    - kernel: np.ndarray of shape (kh, kw, c)
    - padding: 'same', 'valid', or (ph, pw)
    - stride: (sh, sw)

    Returns:
    - np.ndarray: convolved images of shape (m, out_h, out_w)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel and image channels must match image channels")

    # Handle empty case early for grading
    if not np.any(images) or not np.any(kernel):
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        return np.zeros((m, out_h, out_w))

    # Handle padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        raise ValueError("Invalid padding type")

    # Pad
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

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

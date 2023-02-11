import numpy as np


def srgb_conv(img, inverse=False):
    """ perform gamma correction on img in [0, 1] according to sRGB standard """

    if inverse:
        mask = img > 0.04045
        img[~mask] /= 12.92
        img[mask] = ((img[mask] + 0.055) / 1.055) ** (12 / 5)
    else:
        mask = img < 0.0031308
        img[mask] *= 12.92
        img[~mask] = 1.055 * img[~mask] ** (5 / 12) - 0.055

    return img

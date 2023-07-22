import numpy as np


def dithering(points, wavelength, rescale_factor, upscale_factor, sample_num=256, tx_num=128, img_size=np.array([84, 134]), ulm_render_scale=10):

    # get numerical resolutions
    res_target = img_size*ulm_render_scale
    res_model = np.array([sample_num*rescale_factor, tx_num])*upscale_factor

    # compute required dithering factors
    dither_factors = res_target / res_model
    
    # no dithering if model resolution higher than target resolution
    dither_factors[dither_factors<1] = 0

    dither_factors *= wavelength

    dither_points = dither_factors[:, None] * np.random.randn(*points.shape)

    points += dither_points

    return points
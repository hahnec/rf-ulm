import numpy as np


def dithering(points, ulm_render_scale, rescale_factor, upscale_factor, sample_num=256, tx_num=128, img_size=np.array([84, 134])):

    # get numerical resolutions
    res_target = img_size*ulm_render_scale
    res_model = np.array([sample_num*rescale_factor, tx_num])*upscale_factor

    # compute required dithering factors
    dither_factors = res_target / res_model
    
    # no dithering if model resolution higher than target resolution
    dither_factors[dither_factors<1] = 0
    dither_factors /= ulm_render_scale

    rand_nums = np.random.rand(*points.shape)-.5    # [-.5, .5] range
    dither_points = dither_factors[:, None] * rand_nums

    points += dither_points

    return points
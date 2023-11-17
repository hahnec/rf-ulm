import numpy as np
import cv2

from utils.dithering import dithering


def render_ulm_frame(all_pts, imgs, img_size, cfg, fps, scale=None, interpol_method=0):
    
    scale = 10 if scale is None else scale

    # for point dimension consistency
    all_pts = [p[:, :2] for p in all_pts if p.size > 0]
    
    # keep original variable
    ref_size = img_size.copy()

    # consider RF-based point density
    if cfg.model == 'sgspcn' and cfg.skip_bmode and not cfg.dither and interpol_method > 0:
        s = 128/ref_size[1]
        t = 256/ref_size[0] if interpol_method == 2 else 1
        all_pts = [np.array([p[:, 0]*s, p[:, 1]*t]).T for p in all_pts if p.size > 0]
        old_size = ref_size.copy()
        ref_size[1] = 128
        if interpol_method == 2: ref_size[0] = 256

    if cfg.dither:
        # dithering
        img_shape = np.array(imgs[0].shape[-2:])[::-1] if cfg.input_type == 'rf' else ref_size
        y_factor, x_factor = img_shape / ref_size
        all_pts = dithering(all_pts, cfg.upscale_factor, cfg.upscale_factor, x_factor, y_factor)

    if cfg.upscale_factor < scale and not cfg.dither:
        sres_ulm_img, velo_ulm_img = tracks2img(all_pts, img_size=ref_size, scale=cfg.upscale_factor, mode=cfg.track, fps=fps)
        if cfg.upscale_factor != 1:
            sres_ulm_img = cv2.resize(sres_ulm_img, scale*ref_size[::-1], interpolation=cv2.INTER_CUBIC)
            sres_ulm_img[sres_ulm_img<0] = 0
    else:
        sres_ulm_img, velo_ulm_img = tracks2img(all_pts, img_size=ref_size, scale=scale, mode=cfg.track, fps=fps)

    if ref_size[1] == 128:
        if ref_size[0] == 256:
            # 2-D interpolation
            sres_ulm_img = cv2.resize(sres_ulm_img, scale*old_size[::-1], interpolation=cv2.INTER_CUBIC)
            sres_ulm_img[sres_ulm_img<0] = 0
        else:
            # horizontal interpolation only
            x = np.arange(sres_ulm_img.shape[1])
            interp_func = interpolate.interp1d(x, sres_ulm_img, kind='linear', axis=1, fill_value='extrapolate')
            new_x = np.linspace(0, sres_ulm_img.shape[1] - 1, old_size[1]*scale)
            sres_ulm_img = interp_func(new_x)

    return sres_ulm_img, velo_ulm_img


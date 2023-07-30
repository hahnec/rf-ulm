import numpy as np
import scipy


save_tmats = lambda tmats, name=None: np.save('t_mats.npy', tmats) if name is None else np.save(name + '.npy', tmats)


def generate_pala_points_and_samples(cfg, point_num=1e3):

    # generate synthetic points
    synth_points = 2*np.random.rand(2, int(point_num))-1    # [-1, +1] range

    # scale points to PALA range (slightly overshoot for better fitting)
    synth_points[0, ...] *= 80
    synth_points[1, ...] += 1
    synth_points[1, ...] *= 60

    # prepare PALA parameters for projection
    from datasets.pala_dataset.pala_rf import PalaDatasetRf
    dataset = PalaDatasetRf(
        dataset_path=cfg.data_dir,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
    )
    wavelength = dataset.get_key('wavelength')
    
    # project samples
    synth_samples = dataset.project_points_toa_compound(synth_points * wavelength)

    return synth_samples, synth_points


def get_inverse_mapping(cfg, p=6, weights_opt=True, point_num=1e3):

    synth_samples, synth_points = generate_pala_points_and_samples(cfg, point_num)

    synth_samples = synth_samples.swapaxes(1, 2)

    t_mats = get_samples2points_mapping(synth_samples, synth_points, channel_num=128, upscale_factor=cfg.upscale_factor, p=p, weights_opt=weights_opt)

    return t_mats


def get_samples2points_mapping(samples, points, channel_num=128, upscale_factor=4, p=6, weights_opt=False):

    # choose earliest arriving sample positions for each target 
    rf_pts = np.array([samples.min(1), samples.argmin(1)])

    # accout for point indices outside original transducer width
    xe_num = samples.shape[1]
    border_index = (xe_num-channel_num*upscale_factor)//2
    rf_pts[1, ...] -= border_index

    # homogenize B-mode points
    bmode_pts = np.array([*points, np.ones(points.shape[-1])])

    # compute weights
    if weights_opt:
        bmode_mean = np.mean(points, axis=-1)[:, None]
        bmode_dist = ((points - bmode_mean)**2).sum(0)**.5
        weights = bmode_dist[None, :] / bmode_dist.max()
    else:
        weights = np.ones(points.shape[-1])

    t_mats = []
    for wv_idx in range(3):
        # homogenize points
        sample_pts = np.array([*rf_pts[::-1, wv_idx], np.ones(rf_pts.shape[-1])])

        # objective function
        obj_fun = lambda x, a=sample_pts, b=bmode_pts, w=weights: (w*(np.concatenate([x, np.eye(3).flatten()[p:]]).reshape(3, 3) @ a - b)**2).sum()

        # fit affine map (4) + translation (2) from 6 parameters
        x = scipy.optimize.minimize(obj_fun, x0=np.eye(3).flatten()[:p]).x

        # construct final matrix
        t_mat = np.concatenate([x, np.eye(3).flatten()[p:]]).reshape(3, 3)

        t_mats.append(t_mat)

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(*bmode_pts[:2, :], 'rx')
            plt.plot(*(t_mat @ sample_pts)[:2, :], 'b+')
            plt.show()

    return np.stack(t_mats)

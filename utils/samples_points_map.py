import numpy as np
import scipy


save_tmats = lambda tmats, name=None: np.save('t_mats.npy', tmats) if name is None else np.save(name + '.npy', tmats)


def generate_pala_points(cfg, point_num=1e3):

    # generate synthetic points
    synth_points = 2*np.random.rand(2, int(point_num))-1    # [-1, +1] range

    # scale points to PALA range
    synth_points[0, ...] *= 40
    synth_points[1, ...] += 1
    synth_points[1, ...] *= 40

    return synth_points


def get_inverse_mapping(dataset=None, channel_num=128, p=6, weights_opt=False, point_num=1e3):

    synth_points = generate_pala_points(point_num)
    
    # prepare PALA parameters for projection
    if dataset is None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load('./config.yml')
        from datasets.pala_dataset.pala_rf import PalaDatasetRf
        dataset = PalaDatasetRf(
            dataset_path=cfg.data_dir,
            rescale_factor = cfg.rescale_factor,
            upscale_factor = cfg.upscale_factor,
        )
    wavelength = dataset.get_key('wavelength')
    
    # project points to samples
    synth_samples = dataset.project_points_toa_compound(synth_points * wavelength, interpol=True, extrapol=True)
    synth_samples = synth_samples.swapaxes(1, 2) * dataset.upscale_factor

    t_mats = get_samples2points_mapping(synth_samples, synth_points, channel_num=channel_num*dataset.upscale_factor, p=p, weights_opt=weights_opt)

    return t_mats


def get_samples2points_mapping(samples, points, channel_num=128, p=6, weights_opt=False):

    # choose earliest arriving sample positions for each target 
    rf_pts = np.array([samples.min(1), samples.argmin(1)])

    # accout for point indices outside original transducer width
    xe_num = samples.shape[1]
    border_index = (xe_num-channel_num)//2
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

    def obj_fun(x, a, b=bmode_pts, w=weights, p=p, affine=True):

        if not affine:
            x[1] = 0
            x[3] = 0

        a_mat = np.concatenate([x, np.eye(3).flatten()[p:]]).reshape(3, 3)

        pts = a_mat @ a

        if p > 6: pts /= pts[-1, :]

        loss = (w*(pts - b)**2).sum()

        return loss

    t_mats = []
    for wv_idx in range(rf_pts.shape[1]):
        # homogenize points
        sample_pts = np.array([*rf_pts[::-1, wv_idx], np.ones(rf_pts.shape[-1])])

        # objective function with passed arguments
        obj_fun_args = lambda x, a=sample_pts: obj_fun(x, a)

        # fit affine map (4) + translation (2) from 6 parameters
        x = scipy.optimize.minimize(obj_fun_args, x0=np.eye(3).flatten()[:p]).x

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

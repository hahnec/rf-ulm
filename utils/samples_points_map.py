import numpy as np
import scipy


save_tmats = lambda tmats, name=None: np.save('t_mats.npy', tmats) if name is None else np.save(name + '.npy', tmats)


def get_samples2points_mapping(samples, points, p=6, weights_opt=False):

    # choose earliest arriving sample positions for each target 
    rf_pts = np.array([samples.min(1), samples.argmin(1)])

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

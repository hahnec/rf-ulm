import torch
import numpy as np
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.pala_error import rmse_unique

from utils.dithering import dithering


def align_points(masks, gt_pts, t_mat, cfg, sr_img=None):
    
    # gt points alignment
    gt_points = []
    for batch_gt_pts in gt_pts:
        pts_gt = batch_gt_pts[~(torch.isnan(batch_gt_pts.squeeze()).sum(-1) > 0)].numpy()[:, ::-1]
        pts_gt = pts_gt.swapaxes(-2, -1)
        pts_gt = np.fliplr(pts_gt)
        if cfg.input_type == 'rf': pts_gt /= cfg.wavelength
        gt_points.append(pts_gt)

    # extract indices from predicted map
    es_indices = torch.nonzero(masks.squeeze(1)).double()
    es_indices = es_indices.cpu().numpy()

    # apply radial symmetry
    if cfg.radial_sym_opt and sr_img is not None: 
        es_indices[:, 1:] = radial_pala(sr_img.cpu().numpy(), es_indices[:, 1:].astype('int'), w=2)

    # estimated points alignment
    es_points = []
    for i in range(cfg.batch_size):
        if cfg.input_type == 'rf':
            #es_pts = np.vstack([es_indices[es_indices[:, 0]==i, :][:, 1:].T, np.ones(es_indices.shape[0])])
            es_pts = np.fliplr(es_indices[es_indices[:, 0]==i, :]).T
            es_pts[2] = 1
            es_pts[0, :] /= cfg.upscale_factor
            es_pts = t_mat @ es_pts
            es_pts[:2, :] /= cfg.wavelength
        if cfg.input_type == 'iq':
            es_pts = es_indices[es_indices[:, 0]==i, 1:].T
            es_pts /= cfg.upscale_factor
        es_pts = es_pts[:2, ...]

        # dithering
        if cfg.dither:
            es_pts = dithering(es_pts, 10, rescale_factor=cfg.rescale_factor, upscale_factor=cfg.upscale_factor)

        es_points.append(es_pts)

    return es_points, gt_points


def get_pala_error(es_points: np.ndarray, gt_points: np.ndarray, tol=1/4):

    results = []
    for es_pts, gt_pts in zip(es_points, gt_points):
        if gt_pts.size == 0:
            continue

        result = rmse_unique(es_pts.T, gt_pts.T, tol=tol)
        results.append(result)

    return results

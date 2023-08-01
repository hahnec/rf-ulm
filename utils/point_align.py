import torch
import numpy as np
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.pala_error import rmse_unique


def align_points(masks, gt_pts, t_mat, cfg, sr_img=None):
    
    # gt points alignment
    gt_points = []
    for batch_gt_pts in gt_pts:
        nan_mask = torch.isnan(batch_gt_pts.squeeze()).sum(-1) > 0
        gt_rearranged = np.array(batch_gt_pts[~nan_mask].T)[:, ::-1] - np.array([[cfg.origin_x], [cfg.origin_z]])
        gt_points.append(gt_rearranged)

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
            pts = es_indices[es_indices[:, 0]==i, :]
            es_pts = np.vstack([pts[:, 1], pts[:, 2], np.ones(len(pts[:, 2]))])
            es_pts[1, :] /= cfg.upscale_factor
            es_pts = (t_mat @ es_pts)[:2, :]
            es_pts -= np.array([[cfg.origin_x], [cfg.origin_z]])
        if cfg.input_type == 'iq':
            es_pts = es_indices[es_indices[:, 0]==i, 1:].T
            es_pts /= cfg.upscale_factor
            es_pts = np.flipud(es_pts)

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

import torch
import numpy as np
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.pala_error import rmse_unique


def align_points(masks, gt_pts, t_mat, cfg, sr_img=None, stretch_opt=False):
    
    # gt points alignment
    gt_points = []
    for batch_gt_pts in gt_pts:
        nan_mask = torch.isnan(batch_gt_pts.squeeze()).sum(-1) > 0
        gt_rearranged = batch_gt_pts[~nan_mask].T if nan_mask.numel() > 1 else batch_gt_pts[nan_mask].T.squeeze(1)
        gt_rearranged = np.array(gt_rearranged)[:, ::-1] - np.array([[cfg.origin_x], [cfg.origin_z]]) if len(gt_rearranged) > 0 else gt_rearranged
        if stretch_opt and len(gt_rearranged)>0: gt_rearranged[1] /= 2
        gt_points.append(gt_rearranged)

    # extract indices from predicted map
    es_indices = torch.nonzero(masks.squeeze(1))
    es_indices = es_indices.double().cpu().numpy()
    confidence = masks.squeeze(1)[es_indices.swapaxes(0, 1)].double().cpu().numpy()[None, :] if es_indices.size > 0 else np.array([])

    # apply radial symmetry
    if cfg.radial_sym_opt and sr_img is not None and len(es_indices) > 0: 
        es_indices[:, 1:] = radial_pala(sr_img.squeeze().cpu().numpy(), es_indices[:, 1:].astype('int'), w=2)

    # estimated points alignment
    es_points = []
    for i in range(cfg.batch_size):
        if cfg.input_type == 'rf' and cfg.skip_bmode:
            pts = es_indices[es_indices[:, 0]==i, :]
            es_pts = np.vstack([pts[:, 1], pts[:, 2], np.ones(len(pts[:, 2]))])
            es_pts = (t_mat @ es_pts)[:2, :]
            if cfg.channel_num==143: es_pts[0] *= (128/143)
            es_pts -= np.array([[cfg.origin_x], [cfg.origin_z]])
        else:
            es_pts = es_indices[es_indices[:, 0]==i, 1:].T
            es_pts /= cfg.upscale_factor
            if stretch_opt: es_pts[0] /= 2
            es_pts = np.flipud(es_pts)
        es_pts = np.vstack([es_pts, confidence[:, es_indices[:, 0]==i]]) if confidence.size > 0 else es_pts

        es_points.append(es_pts)

    return es_points, gt_points


def get_pala_error(es_points: np.ndarray, gt_points: np.ndarray, tol=1/4):

    results = []
    for es_pts, gt_pts in zip(es_points, gt_points):
        if gt_pts.size == 0:
            continue

        result = rmse_unique(es_pts[:2].T, gt_pts[:2].T, tol=tol)
        results.append(result)

    return results

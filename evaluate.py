import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import dice_coeff
from utils.nms_funs import non_max_supp, non_max_supp_torch
from utils.point_align import align_points, get_pala_error
from utils.point_fusion import cluster_points
from utils.threshold import estimate_threshold


@torch.inference_mode()
def evaluate(model, dataloader, amp, cfg, t_mats):

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    if cfg.input_type == 'rf':
        from sklearn.cluster import DBSCAN
        cluster_obj = DBSCAN(eps=cfg.eps, min_samples=1)

    # iterate over the validation set
    with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            
            wv_es_points = []
            for wv_idx in cfg.wv_idcs:
                imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[0][:, wv_idx], batch[1][:, wv_idx], batch[4])
                
                # move images and labels to correct device and type
                imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=cfg.device, dtype=torch.long)

                # predict the mask
                masks_pred = model(imgs)

                # non-maximum suppression
                masks = non_max_supp_torch(masks_pred, size=cfg.nms_size)
                masks[masks < cfg.nms_threshold] = 0
                masks[masks > 0] -= cfg.nms_threshold

                # point alignment
                es_points, gt_points = align_points(masks, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg)
                wv_es_points.append(es_points)

            # point fusion from compounded waves
            if len(cfg.wv_idcs) > 1:
                pts = np.hstack([wv_es_points[1][0], wv_es_points[0][0], wv_es_points[2][0]])
                # fuse points using DBSCAN when eps > 0 and 
                es_points = [cluster_points(pts[:2].T, cluster_obj=cluster_obj).T] if pts.size > 0 and cfg.eps > 0 else [pts]
            else:
                es_points = wv_es_points[0]

            # evaluation metrics
            pala_err_batch = get_pala_error(es_points, gt_points)

            # threshold analysis
            if true_masks[0].sum() > 0 and torch.any(~torch.isnan(masks_pred)):
                threshold = estimate_threshold(true_masks, masks_pred)
            else:
                threshold = float('NaN')
            
            # compute the dice score
            masks = masks > 0
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            dice_score += dice_coeff(masks, true_masks, reduce_batch_first=False)

    model.train()

    return dice_score / max(num_val_batches, 1), pala_err_batch, masks, threshold

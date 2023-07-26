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
def evaluate(model, dataloader, amp, cfg):

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    if cfg.input_type == 'rf':
        from sklearn.cluster import DBSCAN
        cluster_obj = DBSCAN(eps=1/4, min_samples=1)

    try:
        t_mats = np.load(cfg.tmats_name + '.npy') if cfg.input_type == 'rf' else np.zeros((3,3,3))
    except ValueError:
        import time
        time.sleep(1)
        t_mats = np.load(cfg.tmats_name + '.npy')

    # flip matrices to avoid coordinate flipping during inference
    t_mats[:, :2] = t_mats[:, :2][:, ::-1]
    t_mats[:, :2, :2] = t_mats[:, :2, :2][:, :, ::-1]

    # iterate over the validation set
    with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            
            wv_es_points = []
            for wv_idx in cfg.wv_idcs:
                imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, wv_idx], batch[-2][:, wv_idx], batch[1])
                
                # move images and labels to correct device and type
                imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=cfg.device, dtype=torch.long)

                # predict the mask
                masks_pred = model(imgs)

                # non-maximum suppression
                imgs_nms = non_max_supp_torch(masks_pred, size=cfg.nms_size)
                masks = imgs_nms > cfg.nms_threshold

                # point alignment
                es_points, gt_points = align_points(masks, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg)
                wv_es_points.append(es_points)

            # point fusion from compounded waves
            if cfg.input_type == 'rf':
                es_points = [cluster_points(wv_es_points[1][0].T, wv_es_points[0][0].T, wv_es_points[2][0].T, cluster_obj=cluster_obj).T]
            else:
                es_points = wv_es_points[0]

            # evaluation metrics
            pala_err_batch = get_pala_error(es_points, gt_points)

            # threshold analysis
            if true_masks[0].sum() > 0:
                threshold = estimate_threshold(true_masks, masks_pred)
            else:
                threshold = float('NaN')
            
            # compute the dice score
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            dice_score += dice_coeff(masks, true_masks, reduce_batch_first=False)

    model.train()

    return dice_score / max(num_val_batches, 1), pala_err_batch, masks, threshold

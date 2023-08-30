import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

from utils.dice_score import dice_coeff
from utils.nms_funs import non_max_supp_torch
from utils.point_align import align_points, get_pala_error
from utils.point_fusion import cluster_points
from utils.threshold import estimate_threshold


img_norm = lambda x: (x-x.min())/(x.max()-x.min()) if (x.max()-x.min()) != 0 else x


@torch.inference_mode()
def evaluate(model, dataloader, epoch, val_step, criterion, amp, cfg, wb, t_mats):

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    if cfg.input_type == 'rf':
        from sklearn.cluster import DBSCAN
        cluster_obj = DBSCAN(eps=cfg.eps, min_samples=1)

    # iterate over the validation set
    with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            
            loss = 0
            wv_es_points = []
            for wv_idx in cfg.wv_idcs:
                imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[0][:, wv_idx], batch[1][:, wv_idx], batch[4])
                
                # move images and labels to correct device and type
                imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=cfg.device, dtype=torch.long)

                # use batch size (without shuffling) for temporal stacking with new batch size 1
                if cfg.model == 'smv': 
                    imgs = imgs.unsqueeze(0)
                    true_masks = true_masks.sum(0, keepdim=True)

                # predict the mask
                pred_masks = model(imgs)

                # get loss
                loss += criterion(pred_masks.squeeze(1), true_masks.squeeze(1).float())

                # non-maximum suppression
                masks_nms = non_max_supp_torch(pred_masks, size=cfg.nms_size)
                masks_nms[masks_nms < cfg.nms_threshold] = 0
                masks_nms[masks_nms > 0] -= cfg.nms_threshold

                # point alignment
                es_points, gt_points = align_points(masks_nms, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg)
                wv_es_points.append(es_points)

            # point fusion from compounded waves
            if len(cfg.wv_idcs) > 1:
                wv_list = [el for el in [wv_es_points[1][0], wv_es_points[0][0], wv_es_points[2][0]] if el.size > 0]
                if len(wv_list) > 0:
                    # fuse points using DBSCAN when eps > 0 and 
                    pts = np.hstack(wv_list) if len(wv_list) > 1 else wv_list[0]
                    es_points = [cluster_points(pts[:2].T, cluster_obj=cluster_obj).T] if pts.size > 0 and cfg.eps > 0 else [pts]
            else:
                es_points = wv_es_points[0]

            # evaluation metrics
            pala_err_batch = get_pala_error(es_points, gt_points)

            # threshold analysis
            if true_masks[0].sum() > 0 and torch.any(~torch.isnan(pred_masks)):
                threshold = estimate_threshold(true_masks[0], pred_masks[0])
            else:
                threshold = float('NaN')
            
            # compute the dice score
            masks_nms = masks_nms > 0
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            dice_score += dice_coeff(masks_nms, true_masks, reduce_batch_first=False)

            if cfg.logging:
                wb.log({
                    'val_step': val_step,
                    'val_loss': loss.item(),
                    'threshold': threshold,
                    })
            val_step += 1

    val_score = dice_score / max(num_val_batches, 1)
    logging.info('Validation Dice score: {}'.format(val_score))
    if cfg.logging:
        wb.log({
            'epoch': epoch,
            'validation_dice': val_score,
            'images': wandb.Image(imgs[0].cpu() if len(imgs[0].shape) == 2 else imgs[0].sum(0).cpu()),
            'avg_detected': float(masks_nms[0].float().cpu().sum()),
            'pred_max': float(pred_masks[0].float().cpu().max()),
            'masks': {
                'true': wandb.Image(img_norm(true_masks[0].float().cpu())*255),
                'pred': wandb.Image(img_norm(pred_masks[0].float().cpu())*255),
                'nms': wandb.Image(img_norm(masks_nms[0].float().cpu())*255),
                },
            })

    model.train()

    return val_step

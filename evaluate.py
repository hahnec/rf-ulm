import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from utils.dice_score import dice_coeff
from utils.nms_funs import non_max_supp, non_max_supp_torch
from utils.point_align import align_points, get_pala_error


@torch.inference_mode()
def evaluate(net, dataloader, amp, cfg):

    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    wv_idx = 1
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

            imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, wv_idx], batch[-2][:, wv_idx], batch[1])
            
            # move images and labels to correct device and type
            imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=cfg.device, dtype=torch.long)

            # predict the mask
            masks_pred = net(imgs)

            # evaluation metrics
            imgs_nms = non_max_supp_torch(masks_pred, size=cfg.nms_size)
            masks = imgs_nms > cfg.nms_threshold
            es_points, gt_points = align_points(masks, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg)
            pala_err_batch = get_pala_error(es_points, gt_points)

            # threshold analysis
            if true_masks[0].sum() > 0: # no positive samples in y_true are meaningless
                # calculate the g-mean for each threshold
                fpr, tpr, thresholds = roc_curve(true_masks[0].float().cpu().numpy().flatten(), masks_pred[0].float().cpu().numpy().flatten())
                #precision, recall, thresholds = precision_recall_curve(mask_true.float().numpy().flatten(), masks_pred.float().numpy().flatten())
                gmeans = (tpr * (1-fpr))**.5
                th_idx = np.argmax(gmeans)
                threshold = thresholds[th_idx]
            else:
                threshold = float('NaN')
            
            # compute the dice score
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            dice_score += dice_coeff(masks, true_masks, reduce_batch_first=False)

    net.train()

    return dice_score / max(num_val_batches, 1), pala_err_batch, masks, threshold

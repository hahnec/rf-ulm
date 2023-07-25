import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.nms_funs import non_max_supp, non_max_supp_torch
from utils.dithering import dithering
from datasets.pala_dataset.utils.pala_error import rmse_unique
from datasets.pala_dataset.utils.radial_pala import radial_pala


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


def get_pala_error(es_points: np.ndarray, gt_points: np.ndarray):

    results = []
    for es_pts, gt_pts in zip(es_points, gt_points):
        if gt_pts.size == 0:
            continue
        pts_es = es_pts.T
        pts_gt = gt_pts.T

        result = rmse_unique(pts_es, pts_gt, tol=1/4)
        results.append(result)

    return results


@torch.inference_mode()
def evaluate(net, dataloader, amp, cfg):

    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    wv_idx = 1
    name_ext = '_' + str(int(cfg.upscale_factor)) + '_' + str(int(cfg.rescale_factor))
    try:
        t_mats = np.load('./t_mats' + name_ext + '.npy') if cfg.input_type == 'rf' else np.zeros((3,3,3))
    except ValueError:
        import time
        time.sleep(1)
        t_mats = np.load('./t_mats' + name_ext + '.npy')

    # flip matrices to avoid coordinate flipping during inference
    t_mats[:, :2] = t_mats[:, :2][:, ::-1]
    t_mats[:, :2, :2] = t_mats[:, :2, :2][:, :, ::-1]

    # iterate over the validation set
    with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, wv_idx].unsqueeze(1), batch[-2][:, wv_idx].unsqueeze(1), batch[1])
            
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
            
            # dice score
            if cfg.model in ('unet', 'mspcn'):
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                #mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(masks, true_masks, reduce_batch_first=False)
            else:
                assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], true_masks[:, 1:], reduce_batch_first=False)

    net.train()

    return dice_score / max(num_val_batches, 1), pala_err_batch, masks, threshold

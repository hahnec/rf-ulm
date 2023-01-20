import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.pala_error import rmse_unique
from utils.non_max_supp import NonMaxSuppression


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, cfg):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            #image, mask_true = batch['image'], batch['mask']
            image, mask_true, gt_pts = batch[:3]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            masks_pred = net(image)

            if mask_true.sum() > 0: # no positive samples in y_true are meaningless
                fpr, tpr, thresholds = roc_curve(mask_true.float().cpu().numpy().flatten(), masks_pred.float().cpu().numpy().flatten())
                #precision, recall, thresholds = precision_recall_curve(mask_true.float().numpy().flatten(), masks_pred.float().numpy().flatten())

                # calculate the g-mean for each threshold
                gmeans = (tpr * (1-fpr))**.5
                th_idx = np.argmax(gmeans)
                threshold = thresholds[th_idx]
            else:
                threshold = float('NaN')

            imgs_nms = non_max_supp(masks_pred)
            masks_nms = imgs_nms > cfg.nms_threshold

            gt_pts = [gt_pt[~(torch.isnan(gt_pt.squeeze()).sum(-1) > 0), :].numpy()[:, ::-1] for gt_pt in gt_pts]#gt_pts[:, ~(torch.isnan(gt_pts.squeeze()).sum(-1) > 0)].numpy()[:, ::-1]
            pala_err_batch = get_pala_error(masks_nms.cpu().numpy().squeeze(1), gt_pts, rescale_factor=cfg.rescale_factor)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                #mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(masks_nms, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1), pala_err_batch, masks_nms, threshold


def get_pala_error(mask_pred: np.ndarray, gt_points: np.ndarray, rescale_factor: float = 1):

    wavelength = 9.856e-05
    origin = np.array([-72, 16])

    results = []
    for mask, true_frame_pts in zip(mask_pred, gt_points):
        if true_frame_pts.size == 0:
            continue
        pts = (np.array(np.nonzero(mask))[::-1] / rescale_factor + origin[:, None]).T
        pts_gt = true_frame_pts[:, ::-1] / rescale_factor + origin[:, None].T
        result = rmse_unique(pts, pts_gt, tol=1/4)
        results.append(result)

    rmse, precision, recall, jaccard, tp_num, fp_num, fn_num = torch.nanmean(torch.tensor(results), axis=0) if len(results) > 0 else (float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'))

    return rmse, precision, recall, jaccard, tp_num, fp_num, fn_num

def non_max_supp(masks_pred, norm_opt=False):

    nms_imgs = []
    for mask_pred in masks_pred:
        img = mask_pred.detach().squeeze(0).cpu().numpy()
        img = (img-img.min())/(img.max()-img.min()) if norm_opt else img
        nms_obj = NonMaxSuppression(img=img)
        nms_obj.main()
        nms_img = nms_obj.map
        nms_imgs.append(nms_img)
    nms_imgs = torch.tensor(np.array(nms_imgs), device=masks_pred.device).unsqueeze(1).float()

    return nms_imgs


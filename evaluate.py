import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.nms_funs import non_max_supp, non_max_supp_torch
from datasets.pala_dataset.utils.pala_error import rmse_unique
from datasets.pala_dataset.utils.radial_pala import radial_pala


def align_points(masks, gt_pts, t_mat, cfg):
    
    # gt points alignment
    gt_points = []
    for batch_gt_pts in gt_pts:
        pts_gt = batch_gt_pts[~(torch.isnan(batch_gt_pts.squeeze()).sum(-1) > 0)].numpy()[:, ::-1]
        pts_gt = pts_gt.swapaxes(-2, -1)
        pts_gt = np.fliplr(pts_gt)
        pts_gt /= cfg.wavelength
        gt_points.append(pts_gt)

    # estimated points alignment
    es_indices = torch.nonzero(masks.squeeze(1)).double()
    es_points = []
    for i in range(cfg.batch_size):
        es_pts = torch.fliplr(es_indices[es_indices[:, 0]==i, :]).T
        if cfg.input_type == 'rf':
            es_pts[2] = 1
            es_pts[:2, :] = torch.flipud(es_pts[:2, :])
            es_pts[1, :] /= cfg.upscale_factor
            es_pts = t_mat @ es_pts
            es_pts[:2, :] = torch.flipud(es_pts[:2, :])
        es_pts = es_pts[:2, ...] 

        # dithering
        if cfg.dither:
            es_pts = dithering(es_pts, cfg.wavelength/20, rescale_factor=cfg.rescale_factor, upscale_factor=cfg.upscale_factor)

        es_points.append(es_pts.cpu().numpy() / cfg.wavelength)

    return es_points, gt_points


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, cfg):

    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    name_ext = '_' + str(int(cfg.upscale_factor)) + '_' + str(int(cfg.rescale_factor))
    t_mats = torch.tensor(np.load('./t_mats' + name_ext + '.npy')).to(cfg.device)
    wv_idx = 1

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            image, mask_true, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, 1].unsqueeze(1), batch[-2][:, 1].unsqueeze(1), batch[1])
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            masks_pred = net(image)

            if mask_true[0].sum() > 0: # no positive samples in y_true are meaningless
                # calculate the g-mean for each threshold
                fpr, tpr, thresholds = roc_curve(mask_true[0].float().cpu().numpy().flatten(), masks_pred[0].float().cpu().numpy().flatten())
                #precision, recall, thresholds = precision_recall_curve(mask_true.float().numpy().flatten(), masks_pred.float().numpy().flatten())
                gmeans = (tpr * (1-fpr))**.5
                th_idx = np.argmax(gmeans)
                threshold = thresholds[th_idx]
            else:
                threshold = float('NaN')

            imgs_nms = non_max_supp_torch(masks_pred, size=cfg.nms_size)
            masks = imgs_nms > cfg.nms_threshold

            es_points, gt_points = align_points(masks, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg)

            pala_err_batch = get_pala_error(es_points, gt_points, upscale_factor=cfg.upscale_factor)

            if cfg.model in ('unet', 'mspcn'):
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                #mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(masks, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    return dice_score / max(num_val_batches, 1), pala_err_batch, masks, threshold


def get_pala_error(es_points: np.ndarray, gt_points: np.ndarray, upscale_factor: float = 1, sr_img=None, avg_weight_opt=False, radial_sym_opt=False):

    origin = np.array([-72, 16])

    results = []
    for es_pts, gt_pts in zip(es_points, gt_points):
        if gt_pts.size == 0:
            continue
        pts_es = (es_pts / upscale_factor + origin[:, None]).T
        pts_gt = (gt_pts / upscale_factor + origin[:, None]).T

        # do weighting based on super-resolved image
        if not sr_img is None and avg_weight_opt:
            w = 4
            coords = np.stack(np.meshgrid(np.arange(-w, w+1), np.arange(-w, w+1)))
            for i, pt in enumerate(es_pts):
                pt = ((pt-origin) * upscale_factor).astype(int)[::-1]
                shift_coords = pt[:, None, None]+coords
                try:
                    patch = sr_img[pt[0]-w:pt[0]+w+1, pt[1]-w:pt[1]+w+1][None, ...]
                    pt = np.sum(shift_coords*patch/patch.sum(), axis=(-2, -1))
                except ValueError:
                    pass
                es_pts[i] = pt[::-1] / upscale_factor + origin
        
        # apply radial symmetry
        if radial_sym_opt: es_pts[:, :2] = radial_pala(sr_img, es_pts[:, :2], w=2)

        result = rmse_unique(pts_es, pts_gt, tol=1/4)
        results.append(result)

    return results

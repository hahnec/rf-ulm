import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

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
            image, mask_true, gt_points = batch[:3]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            masks_pred = net(image)
            
            # activation followed by non-maximum suppression
            masks_pred = torch.sigmoid(masks_pred)
            masks_nms = non_max_supp(masks_pred, threshold=cfg.nms_threshold)

            pala_err_batch = get_pala_error(masks_nms, gt_points)

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
    return dice_score / max(num_val_batches, 1), pala_err_batch, masks_nms


def get_pala_error(mask_pred, gt_points):

    pred_pts = [torch.nonzero(m) for m in mask_pred.squeeze()]
    #true_pts = [torch.nonzero(m) for m in mask_true.squeeze()]

    wavelength = 9.856e-05
    #origin = torch.tensor([-72, 16], device=mask_pred.device)

    #pred_pts = [(p/8)*wavelength for p in pred_pts]
    #true_pts = [(p/8)*wavelength for p in true_pts]

    results = []
    for pred_frame_pts, true_frame_pts in zip(pred_pts, gt_points):
        result = rmse_unique(pred_frame_pts.cpu().numpy(), true_frame_pts.cpu().numpy(), tol=1/4)
        results.append(result)
    rmse, precision, recall, jaccard, tp_num, fp_num, fn_num = torch.nanmean(torch.tensor(results), axis=0)

    return rmse, precision, recall, jaccard, tp_num, fp_num, fn_num

def non_max_supp(masks_pred, threshold=0.5, norm_opt=False):

    masks_nms = []
    for mask_pred in masks_pred:
        img = mask_pred.detach().squeeze(0).cpu().numpy()
        img = (img-img.min())/(img.max()-img.min()) if norm_opt else img
        nms_obj = NonMaxSuppression(img=img)
        nms_obj.main()
        nms_img = nms_obj.map
        masks_nms.append(nms_img > threshold)
    masks_nms = torch.tensor(np.array(masks_nms), device=masks_pred.device).unsqueeze(1).float()

    return masks_nms

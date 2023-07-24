import torch

from utils.non_max_supp import NonMaxSuppression


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

def non_max_supp_torch(frame, size=3, norm_opt=False):

    mask = torch.nn.functional.max_pool2d(frame.clone(), kernel_size=size, stride=1, padding=size//2)

    mask[mask != frame] = 0

    return mask

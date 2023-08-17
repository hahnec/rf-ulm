import torch


def non_max_supp_torch(frame, size=3, norm_opt=False):

    mask = torch.nn.functional.max_pool2d(frame.clone(), kernel_size=size, stride=1, padding=size//2)

    mask[mask != frame] = 0

    return mask

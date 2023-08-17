import torch


def non_max_supp_torch(frame, size=3, norm_opt=False):

    # ensure odd integer padding
    size = size//2*2 - 1

    # max-pooling
    mask = torch.nn.functional.max_pool2d(frame.clone(), kernel_size=size, stride=1, padding=size//2)

    # suppression of non-maxima
    mask[mask != frame] = 0

    return mask

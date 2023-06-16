import numpy as np
from skimage.morphology import local_maxima


def weighted_avg(img, pts, w=4):

    coords = np.stack(np.meshgrid(np.arange(-w, w+1), np.arange(-w, w+1)))
    for i, pt in enumerate(pts):
        pt = pt.astype(int)
        shift_coords = pt[:, None, None]+coords
        try:
            patch = img[pt[0]-w:pt[0]+w+1, pt[1]-w:pt[1]+w+1][None, ...]
            pts[i] = np.sum(shift_coords*patch/patch.sum(), axis=(-2, -1))
        except ValueError:
            pass

    return pts


def argmax_blob(img, pts, w=4):

    for i, pt in enumerate(pts):
        pt = pt.astype(int)
        try:
            idx = np.argmax(img[pt[0]-w:pt[0]+w+1, pt[1]-w:pt[1]+w+1])
            pts[i] = pt + np.array([idx//(2*w+1)-w, idx%(2*w+1)-w])
        except ValueError:
            pass

    return pts

def regional_max(img_conv, th=None, point_num=None):

    maxima = regional_mask(img_conv, th=th, point_num=point_num)
    yxr_pts = np.array(np.where(maxima>0), dtype=np.float64).T

    return yxr_pts

def regional_mask(img_conv, th=None, point_num=None):

    # return all points if point_num not set
    point_num = -1 if point_num is None else point_num

    maxima = local_maxima(img_conv) * img_conv
    #maxima[:+2, :] = 0
    #maxima[-2:, :] = 0
    #maxima[:, :+2] = 0
    #maxima[:, -2:] = 0
    if th is None:
        th = np.sort(np.unique(maxima))[::-1][point_num-1] if point_num < len(np.unique(maxima)) else np.inf
    maxima[maxima<th] = 0

    return maxima

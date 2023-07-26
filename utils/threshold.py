import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def estimate_threshold(true_masks, masks_pred):

    # calculate the g-mean for each threshold
    fpr, tpr, thresholds = roc_curve(true_masks[0].float().cpu().numpy().flatten(), masks_pred[0].float().cpu().numpy().flatten())
    #precision, recall, thresholds = precision_recall_curve(mask_true.float().numpy().flatten(), masks_pred.float().numpy().flatten())
    gmeans = (tpr * (1-fpr))**.5
    th_idx = np.argmax(gmeans)
    threshold = thresholds[th_idx]

    return threshold

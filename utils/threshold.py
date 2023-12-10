import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def estimate_threshold(true_masks, pred_masks):

    # calculate the g-mean for each threshold
    fpr, tpr, thresholds = roc_curve(true_masks.float().cpu().numpy().flatten(), pred_masks.float().cpu().numpy().flatten())
    #precision, recall, thresholds = precision_recall_curve(mask_true.float().numpy().flatten(), pred_masks.float().numpy().flatten())
    gmeans = (tpr * (1-fpr))**.5
    th_idx = np.argmax(gmeans)
    threshold = thresholds[th_idx]

    if False:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(
            true_masks.float().cpu().numpy().flatten(),
            pred_masks.float().cpu().numpy().flatten(),
            name="micro-average OvR",
            color="darkorange")
        from sklearn.metrics import auc
        a = auc(fpr, tpr)
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),linestyle='-.',color='k')
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
        plt.legend({'AUC for classifier: '+str(a)})
        plt.show()

    return threshold

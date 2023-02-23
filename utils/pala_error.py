import numpy as np

jaccard_index = lambda tp, fn, fp: tp/(fn+tp+fp)

def rmse_unique(pt_array, gt_array, tol=1/4):

    if pt_array.size == 0:
        return float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')

    mask = np.ones(len(pt_array), dtype=bool)
    fn_num = 0

    errs = []
    for gt in gt_array:
        ssd = ((pt_array-gt)**2).sum(-1)**.5
        err, idx = np.min(ssd), np.argmin(ssd)
        if err < tol: #and mask[idx]:
            errs.append(err)
            mask[idx] = False
        else:
            fn_num += 1
    
    fp_num = sum(mask)
    tp_num = len(errs)

    jaccard = jaccard_index(tp_num, fn_num, fp_num) * 100
    precision = tp_num/(fp_num+tp_num) * 100 if fp_num+tp_num != 0 else 0
    recall = tp_num/(fn_num+tp_num) * 100 if fn_num+tp_num != 0 else 0

    rmse = np.nanmean(err) if len(err) > 0 else float('NaN')

    return rmse, precision, recall, jaccard, tp_num, fp_num, fn_num

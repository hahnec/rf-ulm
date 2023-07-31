import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def cluster_points(pts, tol=1/4, cluster_obj=None):

    # init cluster object
    co = DBSCAN(eps=tol, min_samples=1) if cluster_obj is None else cluster_obj

    # remove NaN and Inf entries
    vpts = pts[~np.any(np.isinf(pts) | np.isnan(pts), 1), :]

    # clustering
    cluster_labels = co.fit_predict(vpts)

    # get unique labels while points labeled as -1 are considered outliers
    labels_unique, counts = np.unique(co.labels_[co.labels_ != -1], return_counts=True)

    fused_points = []
    for i, label in enumerate(labels_unique):
        cluster_points = vpts[cluster_labels == label]
        fused_point = np.mean(cluster_points, axis=0)
        fused_points.append(fused_point)

    return np.vstack(fused_points) if len(fused_points) > 0 else np.array(fused_points)

#def spatial_distance_fusion(wv_es_points):
#    from scipy.spatial.distance import cdist
#    dists_0 = cdist(wv_es_points[1][0].T, wv_es_points[0][0].T, metric='euclidean')
#    dists_2 = cdist(wv_es_points[1][0].T, wv_es_points[2][0].T, metric='euclidean')
#    min0, amin0 = np.min(dists_0, 0), np.argmin(dists_0, 0)
#    min2, amin2 = np.min(dists_2, 0), np.argmin(dists_2, 0)
#    tol = 1/4
#    idcs0 = amin0[min0<tol]
#    idcs1 = min0<tol
#    idcs2 = amin2[min2<tol]
#    es_points = [np.stack([wv_es_points[1][0][:, min0<tol], wv_es_points[0][0][:, idcs0], wv_es_points[2][0][:, idcs2]]).mean(0)]


def fuse_points_within_tolerance(*args, tol=1/4):

    all_points = np.vstack(args)

    # Compute pairwise distances between all points from different arrays
    distances = cdist(all_points, all_points)

    # Create a mask for points within the tolerance distance
    mask_within_tolerance = distances <= tol

    # Calculate the mean of points within each cluster
    fused_points = np.array([np.mean(all_points[mask], axis=0) for mask in mask_within_tolerance])

    return fused_points


if __name__ == '__main__':

    arr1 = np.array([(1, 2), (3, 4), (6, 8)])
    arr2 = np.array([(2, 3), (5, 6), (9, 10)])
    arr3 = np.array([(0, 1), (3, 5), (7, 9)])
    tolerance = 4.5
    fused_points = fuse_points_within_tolerance(arr1, arr2, arr3, tol=tolerance)
    print(fused_points)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(*fused_points.T, 'ko')
    plt.plot(*arr1.T, 'rx')
    plt.plot(*arr2.T, 'b+')
    plt.plot(*arr2.T, 'g*')
    plt.show()
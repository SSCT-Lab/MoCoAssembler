import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


grouping_module = tf.load_op_library(os.path.join(BASE_DIR, "knn_grouping.so"))
ops.NoGradient('KnnGroupingANN')


def knn_grouping(global_points, ref_points, k, sqrt=False, omp=False):
    """
    Get the KNN indices for M reference points within N global points
    of B batches. (KDTree based, runs on CPU only)
    
    Args:
        global_points: [B, N, 3], whole point cloud.
        ref_points: [B, M, 3], query points.
        k: int, number of neighbours.
        sqrt: bool, whether to apply sqrt on default squared distance.
        omp: whether to enagle OpenMP multi-thread for accelerating.
    Return:
        indices: [B, M, K], KNN index of M query points of each batch.
        dists: [B, M, K], distenct of each NN to the corresponding ref.
    """
    indices, dists = grouping_module.knn_grouping_ann(
        global_points=global_points, query_points=ref_points, K=k, omp=omp
    )
    if sqrt:
        dists = tf.sqrt(dists)
    return indices, dists  # using tf.gather with batch_dims=1 to gather NNs


if __name__ == "__main__":
    import numpy as np
    g_pts = tf.convert_to_tensor(np.random.uniform(-10, 10, (8, 50000, 3)).astype(np.float32))
    r_pts = tf.convert_to_tensor(np.random.uniform(-10, 10, (8, 4096, 3)).astype(np.float32))
    idx, dist = knn_grouping(g_pts, r_pts, 64, sqrt=True)
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


_ball_grouping_module = tf.load_op_library(os.path.join(BASE_DIR, "ball_grouping.so"))
ops.NoGradient('BallGrouping')


def ball_grouping(global_points, ref_points, k, radius):
    """
    "Randomely" get K points within the given radius of each reference 
    points, the first found point will be copied to match k if there 
    is not enouch points in given radius.

    Args:
        global_points: [B, N, 3], whole point cloud.
        ref_points: [B, M, 3], query points.
        k: int, max sampling points.
        radius: query radius.
    Return:
        indices: [B, M, K], index of selected points.
        valid_num: [B, M], number of unique selected points.
    """
    indices, valid_num = _ball_grouping_module.ball_grouping(
        globals=global_points, queries=ref_points, num=k, radius=radius
    )
    return indices, valid_num  # using tf.gather with batch_dims=1 to gather points.


if __name__ == "__main__":
    import numpy as np
    g_pts = tf.convert_to_tensor(np.random.uniform(-10, 10, (8, 50000, 3)).astype(np.float32))
    r_pts = tf.convert_to_tensor(np.random.uniform(-10, 10, (8, 4096, 3)).astype(np.float32))

    idx, count = ball_grouping(g_pts, r_pts, 64, 2)

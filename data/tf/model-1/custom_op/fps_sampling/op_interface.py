import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


_fps_module = tf.load_op_library(os.path.join(BASE_DIR, "fps_sampling.so"))
ops.NoGradient('FarthestPointSample')


def fps_sampling(global_points, ref_nums):
    """
    Get reference points indices thorough Farthest Point Sampling(FPS) 
    over global points. (runs on GPU only)

    Args:
        global_points: [B, N, 3], whole point cloud.
        ref_nums: sampling number of reference points.
    Return:
        indices: [B, M]
    """
    # batch_size = tf.shape(global_points)[0]
    indices = _fps_module.farthest_point_sample(pc=global_points, num=ref_nums)  # [B, M]
    # indices = tf.expand_dims(indices, axis=2)  # [B, M, 1]
    # batch_index = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, ref_nums, 1])  # [B, M, 1]
    # indices = tf.concat([batch_index, indices], axis=2)  # [B, M, 2(B_index, M_index)]
    return indices  # [B,M], using tf.gather with batch_dims=1 to gather points.


if __name__ == "__main__":
    import numpy as np
    pts = tf.convert_to_tensor(np.random.uniform(-10, 10, (8, 50000, 3)).astype(np.float32))
    idx = fps_sampling(pts, 1024)

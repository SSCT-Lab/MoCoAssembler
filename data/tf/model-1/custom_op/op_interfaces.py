import tensorflow as tf


#! custom_op, compile required
from model.custom_op.fps_sampling.op_interface import fps_sampling
from model.custom_op.ball_grouping.op_interface import ball_grouping
from model.custom_op.knn_grouping.op_interface import knn_grouping


def knn_grouping_tf(global_points, ref_points, k, sort=False, sqrt=True):
    """
    Get the KNN indices for M reference points within N global points
    of B batches. (pure tensorflow implementation, brute force)
    
    Args:
        global_points: [B, N, 3], whole point cloud.
        ref_points: [B, M, 3], query points.
        k: int, number of neighbours.
        sort: bool, whether to sort the KNN according to the distance.
        sqrt: bool, whether to apply sqrt on default squared distance.
    Return:
        indice: [B, M, k], KNN index of M query points of each batch.
        dist: distenct for each KNN to the corresponding ref_point.
    """
    batch_size = tf.shape(ref_points)[0]
    ref_nums = tf.shape(ref_points)[1]

    # calculate the distance matrix
    A2 = tf.reduce_sum(ref_points * ref_points, axis=2, keepdims=True)  # [B, M, 1], x1^2 + y1^2 + z1^2
    B2 = tf.reduce_sum(global_points * global_points, axis=2, keepdims=True)  # [B, N, 1], x2^2 + y2^2 + z2^2
    AB = tf.matmul(ref_points, tf.transpose(global_points, perm=[0, 2, 1]))  # [B, M, N], x1*x2 + y1*y2 + z1*z2
    dist_matrix = A2 - 2*AB + tf.transpose(B2, perm=[0, 2, 1])  # [B, M, N]

    # get top-k indices
    dist, indice = tf.nn.top_k(-dist_matrix, k=k, sorted=sort)  # [B, M, k(indices within N)]
    dist = dist * -1  # the dist_matrix was timed by -1 so top_k will return K nearest
    if sqrt:
        dist = tf.sqrt(dist)
    
    return indice, dist  # using tf.gather with batch_dims=1 to gather points.


def interpolate(unknown_points, known_points, known_features, k=3):
    """
    Interpolate the feature of unknown points using K nearest neighbour. (This 
    method is exactly the same as three_nn interpolation defined in pointnet++ 
    when k=3)

    Args:
        unknown_points: [B, M, 3], xyz coordinate of M unknown points.
        known_points: [B, N, 3], xyz coordinate of N known points.
        known_features: [B, N, C], features of known points.
    Return:
        interpolated_features: [B, M, C], interpolated features.
    """
    nn_indice, nn_dist = knn_grouping(known_points, unknown_points, k, sqrt=True, omp=True)  # [B,M,K]
    nn_features = tf.gather(known_features, nn_indice, batch_dims=1)  # [B,M,K,C]
    
    inverse_dist = 1.0 / (nn_dist + 1e-8)  # closer point get bigger weight
    norm = tf.reduce_sum(inverse_dist, axis=2, keepdims=True)  # [B,M,1]
    weight = inverse_dist / norm  # [B,M,K]

    weighted_nn_features = nn_features * tf.expand_dims(weight, axis=3)  # [B,M,K,C]
    interpolated_features = tf.reduce_sum(weighted_nn_features, axis=2)  # [B,M,C]
    return interpolated_features


if __name__ == "__main__":
    pass
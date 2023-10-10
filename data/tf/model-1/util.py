import numpy as np
import tensorflow as tf
from utils.data_utils import nms_2d, nms_3d, batch_get_box_corners, compute_iou


def get_masked_loss(losses, mask):
    """
    Get masked loss, loss is averaged within each batch and summed
    over all batches.

    Args:
        losses: [B,C], losses value for each channels.
        mask: [B,C], mask matrix.
    Return:
        loss: scalar loss.
    """
    # per-batch
    numerator = tf.reduce_sum(losses * mask, axis=1) # [B,]
    denominator = tf.reduce_sum(mask, axis=1) # [B,]
    
    # per-instance
    # numerator = tf.reduce_sum(losses * mask)  # scalar
    # denominator = tf.reduce_sum(mask)  # scalar

    loss = tf.reduce_sum(numerator / (denominator + 1E-7))  # scalar
    return loss


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute huber loss, the only different between this function and
    "tf.losses.huber" is that we don't averaged over last axis (which
    is useful when we need to calculate masked loss).

    Args:
        y_true: [B,C], groundtruth label.
        y_pred: [B,C], predictions.
        delta: delta value for huber loss.
    return:
        loss: [B,C], huber loss value for each elements.
    """
    abs_error = tf.abs(y_true - y_pred)
    deltaed_error = tf.where(abs_error < delta, abs_error, delta)
    loss = 0.5 * deltaed_error**2 + delta * (abs_error - deltaed_error)
    return loss


def distance_matrix(pc1, pc2, metric="L1"):
    """
    Compute distance matrix of two point cloud.

    Args:
        pc1: [B,N,C], point cloud 1 with N points.
        pc2: [B,M,C], point cloud 2 with M points.
        metric(A,B): metric of distance, support "L1", "L1smooth", "L2",
            "L2squared".
    Return:
        dist_matrix: [B,N,M], distance matrix at specific metric.
    """
    N = tf.shape(pc1)[1]
    M = tf.shape(pc2)[1]

    pc1 = tf.tile(tf.expand_dims(pc1, 2), [1, 1, M, 1])  # [B, N, M, C], M is duplicated axis
    pc2 = tf.tile(tf.expand_dims(pc2, 1), [1, N, 1, 1])  # [B, N, M, C], N is duplicated axis

    metric = metric.lower()
    if metric == "l1":
        dist_matrix = tf.reduce_sum(tf.abs(pc1 - pc2), axis=3)
    elif metric == "l1smooth":
        dist_matrix = tf.reduce_sum(huber_loss(pc1, pc2, delta=1.0), axis=3)
    elif metric == "l2":
        dist_matrix = tf.sqrt(tf.reduce_sum((pc1 - pc2)**2, axis=3) + 1E-7)
    elif metric == "l2squared":
        dist_matrix = tf.reduce_sum((pc1 - pc2)**2, axis=3)
    else:
        raise NotImplementedError(f"dist metric of {metric} is not implemented.")

    return dist_matrix  # [B,N,M]


# ---------------- #
# Loss Calculation #
# ---------------- #

def compute_vote_loss(data_dict, pred_dict):
    # pred value
    pred_vote = pred_dict["vote_xyz"]  # [B,M,3]
    pred_vote = tf.tile(tf.expand_dims(pred_vote, axis=2), (1,1,3,1))  # [B,M,3,3]

    # gt value
    scene_idx = pred_dict["scene_idx"]  # [B,M]
    scene_pts_tile = tf.tile(pred_dict["scene_pts"], (1,1,3))  # [B,M,9]
    gt_mask = tf.gather(data_dict["point_votes_mask"], scene_idx, batch_dims=1)  # [B,M,9]
    gt_vote = scene_pts_tile + tf.gather(data_dict["point_votes"], scene_idx, batch_dims=1)  # [B,M,9]
    gt_vote = tf.reshape(gt_vote, (tf.shape(gt_vote)[0], tf.shape(gt_vote)[1], 3, 3))  # [B,M,3,3]
    
    # vote loss
    vote_dist = tf.reduce_sum(tf.abs(pred_vote - gt_vote), axis=3)  # [B,M,3]
    vote_loss = tf.reduce_min(vote_dist, axis=2)  # [B,M]
    vote_loss = get_masked_loss(vote_loss, gt_mask)  # scalar

    return vote_loss


def compute_objectness_loss(data_dict, pred_dict, near_threshold, far_threshold):
    # pred values
    pred_objectness = pred_dict["objectness"]  # [B,P,2]
    pred_base_center = pred_dict["base_center"]  # [B,P,3]
    gt_box_center = data_dict["box_center"]  # [B,I,3]
    gt_box_mask = data_dict["box_mask"]  # [B,I]

    # for each pred, find the nearest gt as the pred target
    pred_gt_dist = distance_matrix(pred_base_center, gt_box_center, metric="L2")  # [B,P,I]
    gt_box_mask_ext = tf.tile(tf.expand_dims(gt_box_mask, axis=1), (1, tf.shape(pred_base_center)[1], 1))  # [B,P,I]
    pred_gt_dist = tf.where(gt_box_mask_ext > 0, pred_gt_dist, 1E10)  # "trying to" masked out padding GT

    match_idx = tf.argmin(pred_gt_dist, axis=2)  # [B,P], closest GT idx for each pred
    match_dist = tf.reduce_min(pred_gt_dist, axis=2)  # [B,P], closest GT dist for each pred
    match_mask = tf.gather(gt_box_mask, match_idx, batch_dims=1)  # [B,P], incase there is no GT instance

    # collect GT values
    gt_objectness = match_mask * tf.where(match_dist < near_threshold, 1.0, 0.0)  # [B,P]
    gt_mask = tf.where(tf.logical_or(match_dist < near_threshold, match_dist > far_threshold), 1.0, 0.0)  # [B,P]

    # objectness_loss
    weight = tf.where(gt_objectness > 0, 0.8, 0.2)  # larger weight on positive GT
    losses = tf.losses.sparse_categorical_crossentropy(gt_objectness, pred_objectness, from_logits=True) * weight  # [B,P]
    objectness_loss = get_masked_loss(losses, gt_mask)  # scalar

    return objectness_loss, match_idx, gt_objectness


def compute_proposal_loss(data_dict, pred_dict, match_idx, match_mask, num_heading_bin, num_class, mean_size):
    # center loss
    pred_center = pred_dict["center"]  # [B,P,3]
    pred_center_mask = match_mask  # [B,P]. gt_objectness from "compute_objectness_loss"
    gt_center = data_dict["box_center"]  # [B,I,3]
    gt_center_mask = data_dict["box_mask"]  # [B,I]
    pred_gt_dist = distance_matrix(pred_center, gt_center, metric="L2squared")  # [B,P,I]
    pred_center_loss = get_masked_loss(tf.reduce_min(pred_gt_dist, axis=2), pred_center_mask)
    gt_center_loss = get_masked_loss(tf.reduce_min(pred_gt_dist, axis=1), gt_center_mask)
    center_loss = pred_center_loss + gt_center_loss

    # heading loss
    angle_per_bin = 2*np.pi / num_heading_bin
    gt_box_heading = tf.math.mod(data_dict["box_heading"], 2*np.pi)  # [B,I]
    gt_box_heading_shifted = tf.math.mod(gt_box_heading + angle_per_bin / 2, 2*np.pi)
    gt_box_bin = tf.floor(gt_box_heading_shifted / angle_per_bin)  # [B,I]
    gt_box_residual = gt_box_heading_shifted - (gt_box_bin * angle_per_bin + angle_per_bin / 2)  # [B,I]

    gt_bin = tf.gather(gt_box_bin, match_idx, batch_dims=1)  # [B,P]
    pred_bin_prob = pred_dict["heading_bin_prob"]  # [B,P,num_heading_bin]
    heading_bin_loss = tf.losses.sparse_categorical_crossentropy(gt_bin, pred_bin_prob, from_logits=True)  # [B,P]
    heading_bin_loss = get_masked_loss(heading_bin_loss, match_mask)  # scalar

    gt_residual = tf.gather(gt_box_residual, match_idx, batch_dims=1)  # [B,P]
    # pred_residual = pred_dict["heading_residual"]  # [B,P,num_heading_bin]
    # gt_bin_onehot = tf.one_hot(tf.cast(gt_bin, tf.int32), num_heading_bin, dtype=tf.float32)  # [B,P,num_heading_bin]
    # pred_residual = tf.reduce_sum(pred_residual * gt_bin_onehot, axis=2)  # [B,P], so non-GT prediction get zero gradient
    # heading_residual_loss = huber_loss(gt_residual, pred_residual)  # [B,P]
    # heading_residual_loss = get_masked_loss(heading_residual_loss, match_mask)  # scalar
    gt_residual_normalized = gt_residual / (angle_per_bin / 2)
    gt_bin_onehot = tf.one_hot(tf.cast(gt_bin, tf.int32), num_heading_bin, dtype=tf.float32)  # [B,P,num_heading_bin]
    pred_residual_normalized = pred_dict["heading_residual_normalized"]  # [B,P,num_heading_bin]
    pred_residual_normalized = tf.reduce_sum(pred_residual_normalized * gt_bin_onehot, axis=2)  # [B,P], so non-GT prediction get zero gradient
    heading_residual_loss = huber_loss(gt_residual_normalized, pred_residual_normalized)  # [B,P]
    heading_residual_loss = get_masked_loss(heading_residual_loss, match_mask)  # scalar

    # size loss
    gt_size_class = tf.gather(data_dict["box_label"], match_idx, batch_dims=1)  # [B,P]
    pred_size_class_prob = pred_dict["size_score"]  # [B,P,num_class]
    size_class_loss = tf.losses.sparse_categorical_crossentropy(gt_size_class, pred_size_class_prob, from_logits=True)  # [B,P]
    size_class_loss = get_masked_loss(size_class_loss, match_mask)  # scalar

    gt_size = tf.gather(data_dict["box_size"], match_idx, batch_dims=1)  # [B,P,3]
    gt_mean_size = tf.gather(mean_size, gt_size_class)  # [B,P,3]
    gt_size_residual = gt_size - gt_mean_size  # [B,P,3]
    # pred_size_residual = pred_dict["size_residual"]  # [B,P,num_class,3]
    # gt_size_class_onehot = tf.expand_dims(tf.one_hot(gt_size_class, num_class, dtype=tf.float32), 3)  # [B,P,num_class,1]
    # pred_size_residual = tf.reduce_sum(pred_size_residual * gt_size_class_onehot, axis=2)  # [B,P,3], make non-GT prediction get zero gradient
    # size_residual_loss = tf.losses.huber(gt_size_residual, pred_size_residual, delta=1)  # [B,P]
    # size_residual_loss = get_masked_loss(size_residual_loss, match_mask)  # scalar
    gt_size_residual_normalized = gt_size_residual / gt_mean_size
    gt_size_class_onehot = tf.expand_dims(tf.one_hot(gt_size_class, num_class, dtype=tf.float32), 3)  # [B,P,num_class,1]
    pred_size_residual_normalized = pred_dict["size_residual_normalized"]  # [B,P,num_class,3]
    pred_size_residual_normalized = tf.reduce_sum(pred_size_residual_normalized * gt_size_class_onehot, axis=2)  # [B,P,3], make non-GT prediction get zero gradient
    size_residual_loss = tf.losses.huber(gt_size_residual_normalized, pred_size_residual_normalized, delta=1)  # [B,P]
    size_residual_loss = get_masked_loss(size_residual_loss, match_mask)  # scalar

    #score loss
    gt_class = gt_size_class  # [B,P]
    pred_class_prob = pred_dict["scores"]  # [B,P,num_class]
    score_loss = tf.losses.sparse_categorical_crossentropy(gt_class, pred_class_prob, from_logits=True)  # [B,P]
    score_loss = get_masked_loss(score_loss, match_mask)

    return center_loss, heading_bin_loss, heading_residual_loss, size_class_loss, size_residual_loss, score_loss


def get_losses(data_dict, pred_dict, near_threshold, far_threshold, num_heading_bin, num_class, mean_size):
    vote_loss = compute_vote_loss(data_dict, pred_dict)

    objectness_loss, match_idx, match_mask = compute_objectness_loss(data_dict, pred_dict, near_threshold, far_threshold)

    center_loss, heading_bin_loss, heading_residual_loss, size_class_loss, size_residual_loss, score_loss = \
        compute_proposal_loss(data_dict, pred_dict, match_idx, match_mask, num_heading_bin, num_class, mean_size)

    box_loss = center_loss + \
               0.1 * heading_bin_loss + heading_residual_loss + \
               0.1 * size_class_loss + size_residual_loss + \
               0.1 * score_loss

    losses = vote_loss + 0.5*objectness_loss + box_loss

    losses = losses * 10

    return losses, vote_loss, objectness_loss, box_loss


# ---------------- #
# Prediction Utils #
# ---------------- #

def parse_prediction(pred_dict, num_heading_bin, mean_size, parse_heading=True):
    # collect raw predictions
    objectness = pred_dict["objectness"]  # [B,P,2]
    center = pred_dict["center"]  # [B,P,3]
    heading_bin_prob = pred_dict["heading_bin_prob"]  # [B,P,num_heading_bin]
    heading_residual = pred_dict["heading_residual"]  # [B,P,num_class]
    size_score = pred_dict["size_score"]  # [B,P,num_class]
    size_residual = pred_dict["size_residual"]  # [B,P,num_class,3]
    sem_score = pred_dict["scores"]  # [B,P,num_class]

    # shape info
    B = tf.shape(sem_score)[0]  # batch size
    P = tf.shape(sem_score)[1]  # num proposal
    C = tf.shape(sem_score)[2]  # num semantic class

    # objectness
    objectness = tf.nn.softmax(objectness, axis=2)  # [B,P,2]
    pred_objectness = objectness[:,:,1]  # [B,P]

    # center
    pred_center = center

    # size
    pred_size_class = tf.argmax(size_score, axis=2)  # [B,P]
    pred_size_base = tf.gather(mean_size, pred_size_class)  # [B,P,3]
    pred_size_residual = tf.gather(size_residual, pred_size_class, batch_dims=2)  # [B,P,3]
    pred_size = pred_size_base + pred_size_residual

    # heading
    if parse_heading:
        pred_heading_bin = tf.argmax(heading_bin_prob, axis=2)  # [B,P]
        angle_per_bin = 2*np.pi / num_heading_bin
        pred_heading_base = tf.cast(pred_heading_bin, tf.float32) * angle_per_bin
        pred_heading_residual = tf.gather(heading_residual, pred_heading_bin, batch_dims=2)  # [B,P]
        pred_heading = pred_heading_base + pred_heading_residual  # [B,P]
    else:
        pred_heading = tf.zeros((B,P), dtype=tf.float32)  # [B,P]

    # semantic label
    pred_label = tf.argmax(sem_score, axis=2)  # [B,P]

    return pred_objectness, pred_center, pred_size, pred_heading, pred_label


def pack_prediction(objectness, center, size, heading, label, class_probs=None, *,
                    nms_type=1, inclass_nms=True, iou_threshold=0.25, objectness_threshold=0.05, keep_all_classes=False):

    # convert TF tensor to numpy array
    objectness = objectness.numpy()  # [B,P]
    center = center.numpy()  # [B,P,3]
    size = size.numpy()  # [B,P,3]
    heading = heading.numpy()  # [B,P]
    label = label.numpy()  # [B,P]
    class_probs = class_probs.numpy() if class_probs is not None else None  # [B,P,num_class]

    # get bounding box corners for NMS
    batch_size = center.shape[0]
    num_proposal = center.shape[1]
    corners = batch_get_box_corners(center, size, heading)  # [B,P,8,3]
    corners_min = np.min(corners, axis=2)  # [B,P,3]
    corners_max = np.max(corners, axis=2)  # [B,P,3]

    # NMS
    nms_mask = np.zeros([batch_size, num_proposal], dtype=np.int32)  # [B,P]
    if nms_type == 0:  # NMS 2D
        for i in range(batch_size):
            corners_2d = np.zeros([num_proposal, 4], dtype=np.float32)
            corners_2d[:,0] = corners_min[i,:,0]
            corners_2d[:,1] = corners_min[i,:,1]
            corners_2d[:,2] = corners_max[i,:,0]
            corners_2d[:,3] = corners_max[i,:,1]
            batch_label = label[i,:] if inclass_nms else None
            pick = nms_2d(corners_2d, objectness[i,:], batch_label, iou_threshold)
            if len(pick) > 0:
                nms_mask[i, pick] = 1
    elif nms_type == 1:  # NMS 3D
        for i in range(batch_size):
            corners_3d = np.zeros([num_proposal, 6], dtype=np.float32)
            corners_3d[:,0] = corners_min[i,:,0]
            corners_3d[:,1] = corners_min[i,:,1]
            corners_3d[:,2] = corners_min[i,:,2]
            corners_3d[:,3] = corners_max[i,:,0]
            corners_3d[:,4] = corners_max[i,:,1]
            corners_3d[:,5] = corners_max[i,:,2]
            batch_label = label[i,:] if inclass_nms else None
            pick = nms_3d(corners_3d, objectness[i,:], batch_label, iou_threshold)
            if len(pick) > 0:
                nms_mask[i, pick] = 1

    # filter predict instances
    batch_instances = []
    if not keep_all_classes:  # for each proposal, we assign it to the label with maximum propability
        for b in range(batch_size):
            batch_instances.append([
                # (score: scalar, center: [3,], size: [3,], heading: scalar, label: scalar, corners: [8,3])
                (objectness[b,i], center[b,i], size[b,i], heading[b,i], label[b,i], corners[b,i])
                for i in range(num_proposal) if nms_mask[b,i] == 1 and objectness[b,i] >= objectness_threshold
            ])
    else:  # each proposal is split into <num_class> proposals, which could imporve the mAP but kinda cheaty
        assert class_probs is not None
        num_class = class_probs.shape[2]
        for b in range(batch_size):  # for each batch
            instances = []
            for i in range(num_proposal):
                if nms_mask[b,i] < 1 or objectness[b,i] < objectness_threshold: continue
                for c in range(num_class):
                    instances.append(  # objectness is "apportion" to all classes
                        (class_probs[b,i,c]*objectness[b,i], center[b,i], size[b,i], heading[b,i], c, corners[b,i])
                    )
            batch_instances.append(instances)
    return batch_instances


def pack_groundtruth(center, size, heading, label, mask):
    # convert TF tensor to numpy array
    center = center.numpy()  # [B,I,3]
    size = size.numpy()  # [B,I,3]
    heading = heading.numpy()  # [B,I]
    label = label.numpy()  # [B,I]
    mask = mask.numpy()  # [B,I]

    # compute corners
    corners = batch_get_box_corners(center, size, heading)

    # pack data
    batch_size = center.shape[0]
    num_proposal = center.shape[1]
    batch_instances = []
    for i in range(batch_size):
        batch_instances.append([
            # (center: [3,], size: [3,], heading: scalar, label: scalar, corners: [8,3])
            (center[i,j], size[i,j], heading[i,j], label[i,j], corners[i,j])
            for j in range(num_proposal) if mask[i,j] == 1
        ])

    return batch_instances


def unpack_prediction(pred_pack):
    score, center, size, heading, label, corners = pred_pack
    return score, label, corners


def unpack_groundtruth(gt_pack):
    center, size, heading, label, corners = gt_pack
    return label, corners


def box3d_iou(corners1, corners2):
    iou_3d, iou_2d = compute_iou(corners1, corners2)
    return iou_3d


if __name__ == "__main__":
    NEAR_THRESHOLD = 0.3
    FAR_THRESHOLD = 0.6

    from model.network import DetectModel
    from utils.dataset_utils.ScanNet.dataloader import get_dataset

    def get_mx(*shape):
        # return tf.random.uniform(shape, -30, 30, dtype=tf.float32)
        return tf.convert_to_tensor(np.random.uniform(1, 30, shape).astype(np.float32))

    ds = get_dataset("datasets/ScanNet", max_instances=60, shuffle=True, augment=True, downsample=20000, split="train")
    ds = ds.batch(4, drop_remainder=False)
    dsit = iter(ds)

    num_class = 18
    num_heading_bin = 18
    mean_size = get_mx(18, 3)

    model = DetectModel(num_class=num_class,
                        num_proposal=128,
                        num_heading_bin=num_heading_bin,
                        mean_size=mean_size)

    # test loss func
    data = next(dsit)
    pred1 = model(data["points"][:,:,:3], training=True)
    loss = get_losses(data, pred1, NEAR_THRESHOLD, FAR_THRESHOLD, num_heading_bin, num_class, mean_size)

    # test pack func
    pred2 = model(data["points"][:,:,:3], training=False)
    parsed_pred2 = parse_prediction(pred2, num_heading_bin, mean_size, parse_heading=True)
    pred_pack = pack_prediction(*parsed_pred2, nms_type=1, iou_threshold=0.25, objectness_threshold=0.005)
    gt_pack = pack_groundtruth(data["box_center"], data["box_size"], data["box_heading"], data["box_label"], data["box_mask"])
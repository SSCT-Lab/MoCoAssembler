from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.common.initializer import HeNormal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from utils.lr_scheduler import get_lr_mindspore
from .resnet import resnet50
from .layer import TripletLoss
from .layer import CrossEntropyLabelSmooth


class M2Net(nn.Cell):

    def __init__(self, num_classes, num_apps, last_stride, model_path, model_name, pooling_type, pretrain_choice, cross_mod=False, training=True):
        super().__init__()
        in_planes = 2048
        self.training = training
        resnet = resnet50()

        if pretrained_backbone:
            print(f'Load pretrain from {pretrained_backbone}')
            load_param_into_net(resnet, load_checkpoint(pretrained_backbone))

        self.base = resnet

        self.gap = nn.AvgPool2d(kernel_size=(16, 8), stride=(16, 8))
        self.flat = ops.Flatten()
        self.bottleneck11 = nn.BatchNorm1d(in_planes)
        self.bottleneck11.beta.requires_grad = False
        self.classifier11 = nn.Dense(in_planes, num_classes, has_bias=False,
                                   weight_init=HeNormal(negative_slope=0, mode="fan_out"))
        self.bottleneck12 = nn.BatchNorm1d(in_planes)
        self.bottleneck12.beta.requires_grad = False
        self.classifier12 = nn.Dense(in_planes, num_apps, has_bias=False,
                                   weight_init=HeNormal(negative_slope=0, mode="fan_out"))

        self.cross_mod = cross_mod
        if self.cross_mod:
            print("Loading Cross Modidity Model!")
            self.base2 = resnet
            self.base3 = resnet

            self.bottleneck21 = nn.BatchNorm1d(in_planes)
            self.bottleneck21.beta.requires_grad = False
            self.classifier21 = nn.Dense(in_planes, num_classes, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))
            self.bottleneck22 = nn.BatchNorm1d(in_planes)
            self.bottleneck22.beta.requires_grad = False
            self.classifier22 = nn.Dense(in_planes, num_apps, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))

            self.bottleneck31 = nn.BatchNorm1d(in_planes)
            self.bottleneck31.beta.requires_grad = False
            self.classifier31 = nn.Dense(in_planes, num_classes, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))
            self.bottleneck32 = nn.BatchNorm1d(in_planes)
            self.bottleneck32.beta.requires_grad = False
            self.classifier32 = nn.Dense(in_planes, num_apps, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))

            r = 3
            self.bottleneck41 = nn.BatchNorm1d(in_planes * r)
            self.bottleneck41.beta.requires_grad = False
            self.classifier41 = nn.Dense(in_planes * r, num_classes, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))
            self.bottleneck42 = nn.BatchNorm1d(in_planes * r)
            self.bottleneck42.beta.requires_grad = False
            self.classifier42 = nn.Dense(in_planes * r, num_apps, has_bias=False,
                                    weight_init=HeNormal(negative_slope=0, mode="fan_out"))

    def construct(self, x, x_2, x_3=None):
        """ Forward """
        res_map = self.base(x)
        if self.cross_mod:
            x_2 = self.base2(x_2)
            x_3 = self.base3(x_3)
            x_con = ops.Concat([res_map, x_2, x_3], dim=1)

            global_feat = self.gap(res_map)
            global_feat = self.flat(global_feat)
            global_feat_2 = self.gap(x_2)
            global_feat_2 = self.flat(global_feat_2)
            global_feat_3 = self.gap(x_3) if x_3 is not None else None
            global_feat_3 = self.flat(global_feat_3) if x_3 is not None else None
            global_con_feat = self.gap(x_con)
            global_con_feat = self.flat(x_con)

            feat = self.bottleneck11(global_feat)
            feat_2 = self.bottleneck21(global_feat_2)
            feat_3 = self.bottleneck31(global_feat_3) if x_3 is not None else None
            feat_con = self.bottleneck41(global_con_feat)

            if not self.training:
                return feat, feat_2, feat_3, feat_con
        else:
            global_feat = self.gap(res_map)
            global_feat = self.flat(global_feat)
            feat = self.bottleneck(global_feat)

            if not self.training:
                return feat
        
        score_id = self.classifier11(feat)
        score_app = self.classifier12(self.bottleneck12(feat))
        if self.cross_mod:
            score_2_id = self.classifier21(feat_2)
            score_2_app = self.classifier22(self.bottleneck22(feat_2))
            score_3_id = self.classifier31(feat_3) if x_3 is not None else None
            score_3_app = self.classifier32(self.bottleneck32(feat_3)) if x_3 is not None else None
            score_con_id = self.classifier41(feat_con)
            score_con_app = self.classifier42(self.bottleneck42(feat_con))

            return  score_id, score_app, global_feat,  \
                    score_2_id, score_2_app, global_feat_2, \
                    score_3_id, score_3_app, global_feat_3, \
                    score_con_id, score_con_app, global_con_feat
        else:
            return score_id, score_app, global_feat

    def get_optimizer(self, cfg, criterion, batch_num):
        optimizer = {}
        params = []
        lr = get_lr_mindspore(
            lr_init=cfg.SOLVER.BASE_LR,
            total_epochs=cfg.SOLVER.MAX_EPOCHS,
            steps_per_epoch=batch_num,
            decay_epochs=cfg.SOLVER.STEPS,
            warmup_epoch=cfg.SOLVER.WARMUP_ITERS,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        )
        lr = Tensor(lr)
        adam_group_params = [
            {"params": self.trainable_params(), 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        ]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = nn.SGD(criterion.trainable_params), momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = nn.Adam(adam_group_params, learning_rate=lr)
        return optimizer

class M2NetLoss(nn.Cell):
    """ Combined loss for ReID Strong Baseline model

    Args:
        num_classes: number of classes
        center_loss_weight: weight of Center loss
        crossentropy_loss_weight: weight of CE loss
        feat_dim: number of features for Center loss
        margin: Triplet loss margin
    """
    def __init__(self, num_classes, num_apps, feat_dim=2048):
        super().__init__()
        self.center_loss_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
        self.crossentropy_loss_weight = cfg.SOLVER.APCE_LOSS_WEIGHT

        self.triplet = TripletLoss(cfg.SOLVER.MARGIN)
        self.min_val = Tensor(1e-12, mstype.float32)
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.xent_app = CrossEntropyLabelSmooth(num_classes=num_apps)
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)

    def construct(self, logits, logits_2, labels):
        """ Forward """
        score, feat = logits
        app_score, app = logits_2
        target = labels
        tloss = global_loss(self.triplet, feat, target, self.min_val)[0]
        xloss = self.xent(score, target)
        closs = self.center_criterion(feat, target)

        if cfg.MODEL.APP_CE_LOSS == 'on':
            weight = Tensor([cfg.SOLVER.APCE_LOSS_WEIGHT]).to('cuda')
            loss = self.crossentropy_loss_weight * xloss + tloss + self.center_loss_weight * closs + weight * self.xent_app(app_score, app)

        loss = self.crossentropy_loss_weight * xloss + tloss + self.center_loss_weight * closs
        return loss

def euclidean_dist(x, y, min_val):
    """ Euclidean distance between each pair of vectors from x and y

    x and y are matrices:
    x = (x_1, | x_2 | ... | x_m).T
    y = (y_1, | y_2 | ... | y_n).T

    Where x_i and y_j are vectors. We calculate the distances between each pair x_i and y_j.

    res[i, j] = dict(x_i, y_j)

    res will have the shape [m, n]

    For calculation we use the formula x^2 - 2xy + y^2.

    Clipped to prevent zero distances for numerical stability.
    """
    m, n = x.shape[0], y.shape[0]
    xx = ops.pows(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
    yy = ops.pows(y, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    dist = xx + yy

    dist = 1 * dist - 2 * ops.dot(x, y.transpose())

    # Avoiding zeros for numerical stability
    dist = ops.maximum(
        dist,
        min_val,
    )
    dist = ops.sqrt(dist)
    return dist


def normalize(x, axis=-1):
    norm = ops.sqrt(ops.pows(x, 2).sum(axis=axis, keepdims=True))
    x_normalized = 1. * x / (norm + 1e-12)
    return x_normalized


def hard_example_mining(dist_mat, labels):
    """ Search min negative and max positive distances

    Args:
        dist_mat: distance matrix
        labels: real labels

    Returns:
        distance to positive indices
        distance to negative indices
        positive max distance indices
        negative min distance indices

    """
    def get_max(dist_mat__, idxs, inv=False):
        """ Search max values in distance matrix (min if inv=True) """
        dist_mat_ = dist_mat__.copy()
        if inv:
            dist_mat_ = -dist_mat_
        # fill distances for non-idxs values as min value
        dist_mat_[~idxs] = dist_mat_.min() - 1
        pos_max = dist_mat_.argmax(axis=-1)
        maxes = dist_mat__.take(pos_max, axis=-1).diagonal()
        return pos_max, maxes

    n = dist_mat.shape[0]

    labels_sq = ops.expand_dims(labels, -1).repeat(n, axis=-1)

    # shape [n, n]
    is_pos = ops.equal(labels_sq, labels_sq.T)  # Same id pairs
    is_neg = ops.not_equal(labels_sq, labels_sq.T)  # Different id pairs

    p_inds, dist_ap = get_max(dist_mat, is_pos)  # max distance for positive and corresponding ids
    n_inds, dist_an = get_max(dist_mat, is_neg, inv=True)  # min distance for negative and corresponding ids

    return dist_ap, dist_an, p_inds, n_inds


def global_loss(tri_loss, global_feat, labels, min_val, normalize_feature=False):
    """ Global loss

    Args:
        tri_loss: triplet loss
        global_feat: global features
        labels: real labels
        normalize_feature: flag to normalize features
        min_val: value to cut distance from below
    Returns:
        loss value
        positive max distance indices
        negative min distance indices
        distance to positive indices
        distance to negative indices
        distance matrix
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat, min_val)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
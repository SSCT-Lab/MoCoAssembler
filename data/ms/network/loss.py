import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor

CLIP_VALUE = 1000.
EPS = 1e-7

@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)

@ops.constexpr(reuse_result=True)
def get_pi(dtype=ms.float32):
    return Tensor(math.pi, dtype)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def batch_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[:, :, 0] = x[:, :, 0] - x[:, :, 2] / 2  # top left x
    y[:, :, 1] = x[:, :, 1] - x[:, :, 3] / 2  # top left y
    y[:, :, 2] = x[:, :, 0] + x[:, :, 2] / 2  # bottom right x
    y[:, :, 3] = x[:, :, 1] + x[:, :, 3] / 2  # bottom right y
    return y

def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def batch_box_area(box):
    return (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(box1)
    area2 = box_area(box2)

    expand_size_1 = box2.shape[0]
    expand_size_2 = box1.shape[0]

    box1 = ops.tile(ops.expand_dims(box1, 1), (1, expand_size_1, 1))
    box2 = ops.tile(ops.expand_dims(box2, 0), (expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # inter = ops.minimum(box1[:, None, 2:], box2[None, :, 2:]) - ops.maximum(box1[:, None, :2], box2[None, :, :2])
    inter = ops.minimum(box1[..., 2:], box2[..., 2:]) - ops.maximum(box1[..., :2], box2[..., :2])
    inter = inter.clip(0., None)
    inter = inter[:, :, 0] * inter[:, :, 1]
    # zhy_test
    return inter / (area1[:, None] + area2[None, :] - inter).clip(EPS, None)  # iou = inter / (area1 + area2 - inter)

def batch_box_iou(batch_box1, batch_box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[B, N, 4])
        box2 (Tensor[B, M, 4])
    Returns:
        iou (Tensor[B, N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = batch_box_area(batch_box1)
    area2 = batch_box_area(batch_box2)

    expand_size_1 = batch_box2.shape[1]
    expand_size_2 = batch_box1.shape[1]
    batch_box1 = ops.tile(ops.expand_dims(batch_box1, 2), (1, 1, expand_size_1, 1))
    batch_box2 = ops.tile(ops.expand_dims(batch_box2, 1), (1, expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = ops.minimum(batch_box1[..., 2:], batch_box2[..., 2:]) - \
            ops.maximum(batch_box1[..., :2], batch_box2[..., :2])
    inter = inter.clip(0., None)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]
    # zhy_test
    return inter / (area1[:, :, None] + area2[:, None, :] - inter).clip(EPS, None)  # iou = inter / (area1 + area2 - inter)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        x1, y1, w1, h1 = ops.split(box1, 1, 4)
        x2, y2, w2, h2 = ops.split(box2, 1, 4)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 1, 4)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 1, 4)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1) # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / get_pi(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox_iou_2(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # box1/2, (n, 4) -> (4, n)
    box1, box2 = box1.transpose(1, 0), box2.transpose(1, 0)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:
                return iou # common IoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, gamma=1.5, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):
        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = ops.Sigmoid()(pred) # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)

class BCEWithLogitsLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, bce_weight=None, bce_pos_weight=None, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(weight=bce_weight, pos_weight=bce_pos_weight, reduction="none")
        self.reduction = reduction # default mean
        assert self.loss_fcn.reduction == 'none'  # required to apply FL to each element

    def construct(self, pred, true, mask=None):

        ori_dtype = pred.dtype
        loss = self.loss_fcn(pred.astype(ms.float32), true.astype(ms.float32))

        if mask is not None:
            loss *= mask

        if self.reduction == 'mean':
            if mask is not None:
                return (loss.sum() / mask.astype(loss.dtype).sum().clip(1, None)).astype(ori_dtype)
            return loss.mean().astype(ori_dtype)
        elif self.reduction == 'sum':
            return loss.sum().astype(ori_dtype)
        else:  # 'none'
            return loss.astype(ori_dtype)

class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False

        h = model.hyp  # hyperparameters
        self.hyp_anchor_t = h["anchor_t"]
        self.hyp_box = h['box']
        self.hyp_obj = h['obj']
        self.hyp_cls = h['cls']

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g),\
                             FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

        m = model.model[-1]  # Detect() module
        _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], dtype=ms.float32)

    def construct(self, p, targets):  # predictions, targets
        lcls, lbox, lobj = 0., 0., 0.

        tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
        tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
                                               ops.stop_gradient(indices), ops.stop_gradient(anchors), \
                                               ops.stop_gradient(tmasks)

        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            tobj = ops.zeros(pi.shape[:4], pi.dtype) # target obj

            n = b.shape[0]  # number of targets
            if n:
                _meta_pred = pi[b, a, gj, gi] #gather from (bs,na,h,w,nc)
                pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

                # Regression
                pxy = ops.Sigmoid()(pxy) * 2 - 0.5
                pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
                pbox = ops.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
                # iou = iou * tmask
                # lbox += (1.0 - iou).mean()  # iou loss
                lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum()  # iou loss

                # Objectness
                # iou = ops.Identity()(iou).clip(0, None)
                iou = ops.stop_gradient(iou).clip(0, None)
                if self.sort_obj_iou:
                    _, j = ops.sort(iou)
                    b, a, gj, gi, iou, tmask = b[j], a[j], gj[j], gi[j], iou[j], tmask[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # tobj[b, a, gj, gi] = iou * tmask  # iou ratio
                # tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.identity(iou).clip(0, None)) * tmask  # iou ratio
                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * iou) * tmask  # iou ratio
                
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets

                    t[mnp.arange(n), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE
                    # lcls += self.BCEcls(pcls, t, tmask[:, None]) # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[layer_index]  # obj loss
            if self.autobalance:
                self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / obji.item()

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls

        return loss * bs, ops.identity(ops.stack((lbox, lobj, lcls, loss)))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
        gain = ops.ones(7, ms.int32) # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
            # Matches
            # if nt:
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1)) #.astype(ms.int32)
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1)) #.astype(ms.int32)
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # # Original
            # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
            # offsets = offsets.view(-1, 2)

            # faster,
            tag1, tag2 = ops.identity(j), ops.identity(k)
            tag1, tag2 = ops.tile(tag1[:, None], (1, 2)), ops.tile(tag2[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
            offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy - offsets, ms.int32)
            gij = gij[:]
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
            anch += (anchors[a],)  # anchors
            tcls += (c,)  # class
            tmasks += (mask_m_t,)

        return ops.stack(tcls), \
               ops.stack(tbox), \
               ops.stack(indices), \
               ops.stack(anch), \
               ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask

# class ComputeLoss(nn.Cell):
    # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         self.sort_obj_iou = False

#         h = model.hyp  # hyperparameters
#         self.hyp_anchor_t = h["anchor_t"]
#         self.hyp_box = h['box']
#         self.hyp_obj = h['obj']
#         self.hyp_cls = h['cls']

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h['cls_pw']], ms.float32), gamma=g),\
#                              FocalLoss(bce_pos_weight=Tensor([h['obj_pw']], ms.float32), gamma=g)
#         else:
#             # Define criteria
#             BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['cls_pw']]), ms.float32))
#             BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h['obj_pw']]), ms.float32))

#         m = model.model[-1]  # Detect() module
#         _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
#         self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors

#         self._off = Tensor([
#             [0, 0],
#             [1, 0],
#             [0, 1],
#             [-1, 0],
#             [0, -1],  # j,k,l,m
#             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#         ], dtype=ms.float32)

#     def construct(self, p, targets):  # predictions, targets
#         lcls, lbox, lobj = 0., 0., 0.

#         tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
#         tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
#                                                ops.stop_gradient(indices), ops.stop_gradient(anchors), \
#                                                ops.stop_gradient(tmasks)

#         # Losses
#         for layer_index, pi in enumerate(p):  # layer index, layer predictions
#             tmask = tmasks[layer_index]
#             b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
#             b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
#             tobj = ops.zeros(pi.shape[:4], pi.dtype) # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 _meta_pred = pi[b, a, gj, gi] #gather from (bs,na,h,w,nc)
#                 pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

#                 # Regression
#                 pxy = ops.Sigmoid()(pxy) * 2 - 0.5
#                 pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
#                 pbox = ops.concat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
#                 # iou = iou * tmask
#                 # lbox += (1.0 - iou).mean()  # iou loss
#                 lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum()  # iou loss

#                 # Objectness
#                 iou = ops.Identity()(iou).clip(0, None)
#                 if self.sort_obj_iou:
#                     _, j = ops.sort(iou)
#                     b, a, gj, gi, iou, tmask = b[j], a[j], gj[j], gi[j], iou[j], tmask[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou
#                 # tobj[b, a, gj, gi] = iou * tmask  # iou ratio
#                 tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.identity(iou).clip(0, None)) * tmask  # iou ratio

#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = ops.fill(pcls.dtype, pcls.shape, self.cn) # targets

#                     t[mnp.arange(n), tcls[layer_index]] = self.cp
#                     lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[layer_index]  # obj loss
#             if self.autobalance:
#                 self.balance[layer_index] = self.balance[layer_index] * 0.9999 + 0.0001 / obji.item()

#         if self.autobalance:
#             _balance_ssi = self.balance[self.ssi]
#             self.balance /= _balance_ssi
#         lbox *= self.hyp_box
#         lobj *= self.hyp_obj
#         lcls *= self.hyp_cls
#         bs = p[0].shape[0]  # batch size

#         loss = lbox + lobj + lcls

#         return loss * bs, ops.identity(ops.stack((lbox, lobj, lcls, loss)))

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         targets = targets.view(-1, 6)
#         mask_t = targets[:, 1] >= 0
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
#         gain = ops.ones(7, ms.int32) # normalized to gridspace gain
#         ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
#         ai = ops.cast(ai, targets.dtype)
#         targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

#         g = 0.5  # bias
#         off = ops.cast(self._off, targets.dtype) * g  # offsets

#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

#             # Match targets to anchors
#             t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
#             # Matches
#             # if nt:
#             r = t[..., 4:6] / anchors[:, None]  # wh ratio
#             j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare

#             # t = t[j]  # filter
#             mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
#             t = t.view(-1, 7)

#             # Offsets
#             gxy = t[:, 2:4]  # grid xy
#             gxi = gain[[2, 3]] - gxy  # inverse
#             jk = ops.logical_and((gxy % 1 < g), (gxy > 1)) #.astype(ms.int32)
#             lm = ops.logical_and((gxi % 1 < g), (gxi > 1)) #.astype(ms.int32)
#             j, k = jk[:, 0], jk[:, 1]
#             l, m = lm[:, 0], lm[:, 1]

#             # # Original
#             # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
#             # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
#             # t = t.view(-1, 7)
#             # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
#             # # t = t.repeat((5, 1, 1))[j]
#             # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
#             # offsets = offsets.view(-1, 2)

#             # faster,
#             tag1, tag2 = ops.identity(j), ops.identity(k)
#             tag1, tag2 = ops.tile(tag1[:, None], (1, 2)), ops.tile(tag2[:, None], (1, 2))
#             j_l = ops.logical_or(j, l).astype(ms.int32)
#             k_m = ops.logical_or(k, m).astype(ms.int32)
#             center = ops.ones_like(j_l)
#             j = ops.stack((center, j_l, k_m))
#             t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
#             t = t.view(-1, 7)
#             mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
#             offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
#             offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
#             # offsets_new[0, :, :] = offsets[0, :, :]
#             offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
#             offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
#             offsets = offsets_new
#             offsets = offsets.view(-1, 2)

#             # Define
#             b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
#                                 ops.cast(t[:, 1], ms.int32), \
#                                 t[:, 2:4], \
#                                 t[:, 4:6], \
#                                 ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
#             gij = ops.cast(gxy - offsets, ms.int32)
#             gi, gj = gij[:, 0], gij[:, 1]  # grid indices
#             gi = gi.clip(0, shape[3] - 1)
#             gj = gj.clip(0, shape[2] - 1)


#             # Append
#             indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
#             tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
#             anch += (anchors[a],)  # anchors
#             tcls += (c,)  # class
#             tmasks += (mask_m_t,)

#         return ops.stack(tcls), \
#                ops.stack(tbox), \
#                ops.stack(indices), \
#                ops.stack(anch), \
#                ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask

if __name__ == '__main__':
    # python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml
    #   --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
    import yaml
    from pathlib import Path
    from mindspore import context
    from network.yolo import Model
    from config.args import get_args
    from utils.general import check_file, increment_path, colorstr

    opt = get_args()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    opt.total_batch_size = opt.batch_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    hyp['label_smoothing'] = opt.label_smoothing

    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    cfg = "./config/network_yolov7/yolov7.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    model.hyp = hyp
    model.set_train(True)
    compute_loss = ComputeLoss(model)

    x = Tensor(np.random.randn(2, 3, 160, 160), ms.float32)
    pred = model(x)
    print("pred: ", len(pred))
    # pred, grad = ops.value_and_grad(model, grad_position=0, weights=None)(x)
    # print("pred: ", len(pred), "grad: ", grad.shape)

    targets = Tensor(np.load("targets_bs2.npy"), ms.float32)
    # loss = compute_loss(pred, targets)
    (loss, _), grad = ops.value_and_grad(compute_loss, grad_position=0, weights=None, has_aux=True)(pred, targets)
    print("loss: ", loss)

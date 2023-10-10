import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class LaneCrossEntropyLoss(nn.Layer):
    def __init__(self, ignore_index=255, weights=None, data_format='NCHW'):
        super(LaneCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weights = weights

    def forward(self, logit, label, semantic_weights=None):
        if isinstance(logit,dict):
            logit = logit['seg']
        temp = F.log_softmax(logit, axis=1)
        loss_func = nn.NLLLoss(
            ignore_index=self.ignore_index,
            weight=paddle.to_tensor(self.weights))
        loss = loss_func(temp, label)
        return loss

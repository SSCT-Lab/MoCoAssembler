import paddle
import paddle.nn as nn
import math


class Flatten(nn.Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape([x.shape[0], -1])


# Simple Conv Block
class ConvBlock(nn.Layer):
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.kernel_size = 3
        n = self.kernel_size * self.kernel_size * outdim
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2.0 / float(n))))
        self.C = nn.Conv2D(indim, outdim, self.kernel_size, padding=padding, weight_attr=weight_attr)

        weight_attr = paddle.framework.ParamAttr(
            initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.framework.ParamAttr(
            initializer=nn.initializer.Constant(value=0.0))
        self.BN = nn.BatchNorm2D(outdim, weight_attr=weight_attr, bias_attr=bias_attr)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2D(2)
            self.parametrized_layers.append(self.pool)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Layer):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


def Conv4():
    return ConvNet(4)

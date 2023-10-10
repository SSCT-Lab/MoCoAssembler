import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, Constant
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SiLU(nn.Cell):
    def construct(self, x):
        mul = ops.Mul()
        sigmoid = ops.Sigmoid()
        return mul(x, sigmoid(x))


class SELayer(nn.Cell):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.SequentialCell(
            [
                nn.Dense(oup, _make_divisible(inp // reduction, 8), weight_init=Normal(0.001)),
                SiLU(),
                nn.Dense(_make_divisible(inp // reduction, 8), oup, weight_init=Normal(0.001)),
                nn.Sigmoid()
            ]
        )


    def construct(self, x):
        b, c, pool_x, pool_y = x.shape
        y = ops.AvgPool((pool_x,pool_y))(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        mul = ops.Mul()
        return mul(x, y)


def conv_3x3_bn(inp, oup, stride):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, 3, stride , pad_mode = "pad",padding = 1, has_bias=False, weight_init=Normal(math.sqrt(2/(9*oup)))),
        nn.BatchNorm2d(oup),
        SiLU()
        ]
    )


def conv_1x1_bn(inp, oup):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, 1, 1,padding = 0 , has_bias=False, weight_init=Normal(math.sqrt(2/oup))),
        nn.BatchNorm2d(oup),
        SiLU()
        ]
    )


class MBConv(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.SequentialCell([
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, padding=0, has_bias=False, weight_init=Normal(math.sqrt(2/hidden_dim))),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad_mode = "pad", padding =1, group=hidden_dim, has_bias=False, weight_init=Normal(math.sqrt(2/(9*hidden_dim)))),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1,padding =0, has_bias=False, weight_init=Normal(math.sqrt(2/oup))),
                nn.BatchNorm2d(oup)
                ]
            )
        else:
            self.conv = nn.SequentialCell([
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, pad_mode = "pad",padding =1, has_bias=False, weight_init=Normal(math.sqrt(2/(9*hidden_dim)))),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1,padding = 0, has_bias=False, weight_init=Normal(math.sqrt(2/oup))),
                nn.BatchNorm2d(oup)
                ]
            )

    def construct(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Cell):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.SequentialCell(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.classifier = nn.Dense(output_channel, num_classes, weight_init=Normal(0.001))



    def construct(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = ops.AvgPool(x.shape[-2:])(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x



def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

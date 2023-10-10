from paddle import fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, BatchNorm, Sequential
import numpy as np
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(fluid.dygraph.Layer):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = Pool2D(global_pooling=True, pool_type='avg')
        self.fc = fluid.dygraph.Sequential(
            (Linear(channel, channel // reduction, act='relu')),
            (Linear(channel // reduction, channel, act='sigmoid'))
        )

    def forward(self, x):
        batch, channel, _, __ = x.shape
        y = self.avg_pool(x)
        y = fluid.layers.flatten(y)
        y = self.fc(y)
        y = fluid.layers.reshape(y, shape=(batch, channel, 1, 1))
        return fluid.layers.elementwise_mul(x, y)


class DepthwiseConv(fluid.dygraph.Layer):
    def __init__(self, input_channel, output_channel, filter_size=3, stride=1, relu=False):
        super(DepthwiseConv, self).__init__()
        self.depthwiseConvBN = fluid.dygraph.Sequential(
            Conv2D(input_channel, output_channel, filter_size=filter_size // 2, stride=stride,
                   groups=input_channel),
            BatchNorm(num_channels=output_channel),
        )
        self.relu = relu

    def forward(self, x):
        y = self.depthwiseConvBN(x)
        if self.relu:
            y = fluid.layers.relu(y)
        return y


class GhostModule(fluid.dygraph.Layer):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup/ratio)
        new_channels = init_channels*(ratio-1)
        self.relu = relu

        self.primaryConv = fluid.dygraph.Sequential(
            Conv2D(num_channels=inp, num_filters=init_channels, filter_size=kernel_size, stride=stride,
                   padding=kernel_size//2),
            BatchNorm(num_channels=init_channels)
        )

        self.cheap_operation = fluid.dygraph.Sequential(
            Conv2D(num_channels=init_channels, num_filters=new_channels, filter_size=dw_size, stride=1,
                   padding=dw_size//2, groups=init_channels),
            BatchNorm(num_channels=new_channels)
        )

    def forward(self, x):
        x1 = self.primaryConv(x)
        if self.relu:
            x1 = fluid.layers.relu(x1)
        x2 = self.cheap_operation(x1)
        if self.relu:
            x2 = fluid.layers.relu(x2)
        out = fluid.layers.concat([x1, x2], axis=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(fluid.dygraph.Layer):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()

        self.conv = fluid.dygraph.Sequential(
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            DepthwiseConv(hidden_dim, hidden_dim, kernel_size, stride) if stride == 2 else Sequential(),
            SELayer(hidden_dim) if use_se else Sequential(),
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False)
        )
        if stride == 1 and inp == oup:
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                DepthwiseConv(inp, inp, 3, stride, relu=True),
                Conv2D(num_channels=inp, num_filters=oup, filter_size=1, stride=1, padding=0),
                BatchNorm(num_channels=oup)

            )

    def forward(self, x):
        return fluid.layers.elementwise_add(
            self.conv(x), self.shortcut(x)
        )


class GhostNet(fluid.dygraph.Layer):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = Sequential(
            ('l0', Conv2D(num_channels=3, num_filters=output_channel, filter_size=3, stride=2, padding=1)),
            ('l1', BatchNorm(num_channels=output_channel, act='relu'))
        )

        input_channel = output_channel
        block = GhostBottleneck

        for num, [k, exp_size, c, use_se, s] in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.add_sublayer(name="l{}".format(num+2),
                                sublayer=block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel


        self.features = layers

        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = Sequential(
            Conv2D(num_channels=input_channel, num_filters=output_channel,
                   filter_size=1, stride=1, padding=0),
            BatchNorm(num_channels=output_channel, act='relu'),
            Pool2D(pool_type='avg', global_pooling=True)
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = Sequential(
            Linear(input_channel, output_channel),
            BatchNorm(output_channel, act='relu'),
        )
        self.final = Linear(output_channel, num_classes)
    def forward(self, x, labels):
        x = self.features(x)
        x = self.squeeze(x)
        x = fluid.layers.flatten(x)
        x = self.classifier(x)
        x = fluid.layers.dropout(x, dropout_prob=0.2)
        x = self.final(x)
        outputs = fluid.layers.softmax(x)
        if labels:
            acc = fluid.layers.accuracy(outputs, labels)
            return outputs, acc
        else:
            return outputs


def ghost_net(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


from paddle.fluid.dygraph.base import to_variable

if __name__ == "__main__":
    with fluid.dygraph.guard():
        x = np.random.randn(5, 3, 224, 224).astype('float32')
        x = to_variable(x)
        # net = SELayer(channel=24)
        # net = DepthwiseConv(24, 24, 3, 1)
        net = ghost_net()
        out = net(x, None)
        print(out.shape)
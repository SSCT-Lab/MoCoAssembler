
"""
MindSpore implementation of `DenseNet`.
Refer to: Densely Connected Convolutional Networks
"""

import math
from collections import OrderedDict

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init


__all__ = [
    "DenseNet",
    "densenet100",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201"
]


def _default_cfgs(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
    }


model_cfgs = {
    'densenet100': _default_cfgs(url='densenet100.ckpt'),
    'densenet121': _default_cfgs(url='densenet121.ckpt'),
    'densenet169': _default_cfgs(url=''),
    'densenet201': _default_cfgs(url=''),
    'densenet161': _default_cfgs(url=''),
}


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


class _DenseLayer(nn.Cell):
    """Basic unit of DenseBlock (using bottleneck layer)"""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1x1(num_input_features, bn_size * growth_rate)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(bn_size * growth_rate, growth_rate)

        # nn.Dropout in MindSpore use keep_prob, diff from Pytorch
        self.keep_prob = 1.0 - drop_rate
        self.dropout = nn.Dropout(keep_prob=self.keep_prob)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.keep_prob < 1:
            new_features = self.dropout(new_features)
        return new_features


class _DenseBlock(nn.Cell):
    """DenseBlock. Layers within a block are densely connected."""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()

        self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)
        self.concat = ops.Concat(axis=1)

    def construct(self, init_features):
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = self.concat((features, new_features))
        return features


class _Transition(nn.Cell):
    """Transition layer between two adjacent DenseBlock"""

    def __init__(self, num_input_features, num_output_features, avg_pool=False):
        super(_Transition, self).__init__()
        if avg_pool:
            pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', conv1x1(num_input_features, num_output_features)),
            ('pool', pool_layer)
        ]))

    def construct(self, x):
        x = self.features(x)
        return x


class DenseNet(nn.Cell):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - number of filters in the first Conv2d
        bn_size (int) - multiplicative factor for number of bottleneck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    __constants__ = ['features']

    def __init__(self, growth_rate, block_config, num_init_features=None, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        layers = OrderedDict()

        # first Conv2d
        if num_init_features:
            num_features = num_init_features
            layers['conv0'] = conv7x7(3, num_features, stride=2, padding=3)
            layers['norm0'] = nn.BatchNorm2d(num_features)
            layers['relu0'] = nn.ReLU()
            layers['pool0'] = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        else:

            num_features = growth_rate * 2
            layers['conv0'] = conv3x3(3, num_features, stride=1, padding=1)
            layers['norm0'] = nn.BatchNorm2d(num_features)
            layers['relu0'] = nn.ReLU()

        # DenseBlock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            layers['denseblock%d' % (i + 1)] = block
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                if num_init_features:
                    transition = _Transition(num_features, num_features // 2, avg_pool=False)
                else:
                    transition = _Transition(num_features, num_features // 2, avg_pool=True)
                layers['transition%d' % (i + 1)] = transition
                num_features = num_features // 2

        # final bn+ReLU
        layers['norm5'] = nn.BatchNorm2d(num_features)
        layers['relu5'] = nn.ReLU()

        self.features = nn.SequentialCell(layers)

        # classification layer
        self.mean = ops.ReduceMean()
        self.classifier = nn.Dense(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.HeUniform(math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                                         cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape))

    def construct(self, x):
        x = self.features(x)
        x = self.mean(x, (2, 3))
        x = self.classifier(x)
        return x


def densenet100(**kwargs):
    return DenseNet(growth_rate=12, block_config=(16, 16, 16), **kwargs)


def densenet121(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **kwargs)


def densenet169(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, **kwargs)


def densenet201(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, **kwargs)


def densenet161(**kwargs):
    return DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, **kwargs)


if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor
    from mindspore.train.serialization import load_checkpoint, load_param_into_net

    model = densenet121()
    print(model)
    dummy_input = Tensor(np.random.rand(8, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)

    param_dict = load_checkpoint('densenet121_ascend_v170_imagenet2012_official_cv_top1acc75.54_top5acc92.73.ckpt')
    load_param_into_net(model, param_dict)
    y = model(dummy_input)

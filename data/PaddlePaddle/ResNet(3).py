import paddle.fluid as fluid
import paddle
from paddle.fluid.dygraph import Linear, Conv3D, BatchNorm, Dropout
import math
from functools import partial
import numpy as np

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


# https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/dygraph_cn/Conv3D_cn.html#conv3d
def conv3x3x3(in_planes, out_planes, filter_size=3, stride=1):
    # 3x3x3 convolution with padding
    return Conv3D(
        in_planes,
        out_planes,
        filter_size=3,
        stride=stride,
        padding=1,
        param_attr=fluid.initializer.MSRAInitializer(uniform=False),
        bias_attr=False
    )


def downsample_basic_block(x, planes, stride):
    out = fluid.layers.pool3d(x, pool_size=1, pool_type='avg', pool_stride=stride)
    zero_pads = fluid.layers.zeros(fluid.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).shape())

    out = fluid.dygraph.to_variable(paddle.tensor.concat([out.data, zero_pads], axis=1))

    return out


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv3D(inplanes, planes, filter_size=1,
                            param_attr=fluid.initializer.MSRAInitializer(uniform=False), bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv3D(
            planes, planes, filter_size=3, stride=stride, padding=1,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False), bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv3D(planes, planes * 4, filter_size=1,
                            param_attr=fluid.initializer.MSRAInitializer(uniform=False), bias_attr=False)
        self.bn3 = BatchNorm(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = fluid.layers.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class ResNet(fluid.dygraph.Layer):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = Conv3D(
            3,
            64,
            filter_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            param_attr=fluid.initializer.MSRAInitializer(uniform=False), bias_attr=False)
        self.bn1 = BatchNorm(64)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.last_duration = int(math.ceil(sample_duration / 16))
        self.last_size = int(math.ceil(sample_size / 32))
        self.fc = Linear(512 * block.expansion, num_classes)
        # 添加 Dropout
        self.dropout = Dropout(p=0.5, dropout_implementation='upscale_in_train')

        # 初始化权重函数和偏置
        for m in self.sublayers():

            if isinstance(m, BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype(np.float32))
                m.bias.set_value(np.zeros(m.bias.shape).astype(np.float32))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:

                # sequential实现
                downsample = fluid.dygraph.Sequential(
                    Conv3D(
                        self.inplanes,
                        planes * block.expansion,
                        filter_size=1,
                        stride=stride,
                        param_attr=fluid.initializer.MSRAInitializer(uniform=False), bias_attr=False),
                    BatchNorm(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        x = fluid.layers.pool3d(x, pool_size=(3, 3, 3), pool_type='max',
                                pool_stride=2, pool_padding=1)

        x = self.layer1(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        x = self.layer3(x)
        # x = self.dropout(x)
        x = self.layer4(x)

        x = fluid.layers.pool3d(x, pool_size=(self.last_duration, self.last_size, self.last_size),
                                pool_type='avg', pool_stride=1)

        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])

        # 固定预训练参数不更新(不需要使用时注释掉)
        x.stop_gradient = True

        # 使用dropout(不需要使用时注释掉)
        # x = self.dropout(x)

        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
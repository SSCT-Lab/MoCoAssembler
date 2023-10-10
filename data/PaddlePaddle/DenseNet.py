import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import math
import sys
from paddle.fluid.param_attr import ParamAttr


# DenseBlock = Batch Normalization+ReLU+Conv(3*3)
# 实际上在BN+ReLU+Conv(3*3)之前还加入了BN+ReLU+Conv(1*1)

# 将BN+ReLU+Conv整合为一个类
class BNConvLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 filter_size,
                 stride=1,
                 groups=1,
                 act='relu'):
        super(BNConvLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 批规范化
        self._batch_norm = BatchNorm(self.full_name(), num_channels, act=act)

    def forward(self, inputs):
        y = self._batch_norm(inputs)
        y = self._conv(y)

        return y


# 一个Bottleneck layer
# 事实上还融合了之后的3*3卷积层
class BottleneckLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(BottleneckLayer, self).__init__(name_scope)

        # 真正的bottleneck Layer  = BN + ReLU + 1*1Conv
        # 用4k(growth rate)个filter，将输入的channel降低至4k个
        self.bn_conv1 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters * 4,
            num_channels=num_channels,
            filter_size=1)

        # 经过上方1*1卷积降维后，再进行3*3卷积，大大减少了计算量，输出channel为num_filter个
        # BN + RelU + 3*3Conv
        self.bn_conv2 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_filters * 4,
            filter_size=3)
        self.drop_out_prob = drop_out_prob

    def forward(self, inputs):
        y = self.bn_conv1(inputs)
        y = fluid.layers.dropout(x=y, dropout_prob=self.drop_out_prob)
        y = self.bn_conv2(y)
        y = fluid.layers.dropout(x=y, dropout_prob=self.drop_out_prob)
        return y


# 一个dense block
class DenseBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(DenseBlock, self).__init__(name_scope)

        self.block = BottleneckLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            drop_out_prob=drop_out_prob)

        # 由于Dense Block中 每一层的输入都是之前所有层的输出
        # 每层输出均为growth rate个feature map，因此计算下一层的输入即为上一层的输入加growth_rate(=num_filter)
        self.block_num = block_num
        channels = num_channels + num_filters
        self.convs = []

        # 循环加入dense block中剩余的层，同时更新下一层要输入的channel数
        for i in range(self.block_num - 1):
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                BottleneckLayer(
                    self.full_name(),
                    num_filters=num_filters,
                    num_channels=channels,
                    drop_out_prob=drop_out_prob))
            self.convs.append(conv_block)
            channels = channels + num_filters

        self.out_channel = channels

    def forward(self, inputs):
        layers = []
        layers.append(inputs)
        y = self.block(inputs)
        layers.append(y)
        for conv in self.convs:
            y = paddle.fluid.layers.concat(layers, axis=1)
            y = conv(y)
            layers.append(y)
        y = paddle.fluid.layers.concat(layers, axis=1)
        return y

    # 一个Translation layer


class TransitionLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(TransitionLayer, self).__init__(name_scope)

        # Bottleneck Layer
        # 将上一步dense block 的输出用1*1卷积降维，输出channel为k(=growth rate)个
        self.conv = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            filter_size=1)

        # 池化层
        self.pool2d = Pool2D(
            self.full_name(),
            pool_size=2,
            pool_stride=2,
            pool_type='avg')

        self.dropout_prob = drop_out_prob

    def forward(self, inputs):
        y = self.conv(inputs)
        y = fluid.layers.dropout(x=y, dropout_prob=self.dropout_prob)
        y = self.pool2d(y)

        return y


# 将Dense Block与对应的Translation Layer合并至一起（在该网络模型中一共有三对，最后一个为单独的Dense Block）
class LoopLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(LoopLayer, self).__init__(name_scope)

        # 加入一个dense block
        self.denseblock = DenseBlock(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            block_num=block_num,
            drop_out_prob=drop_out_prob)

        self.channel = self.denseblock.out_channel

        # 加入一个translation layer
        self.transblock = TransitionLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=self.channel,
            drop_out_prob=drop_out_prob)

    def forward(self, inputs):
        y = self.denseblock(inputs)
        y = self.transblock(y)
        return y


# 只使用BottleNeck layer的DenseNet为DenseNet-B，如果要进行进一步压缩，每次translation layer得到m个从dense block中输出的feature map
# 将会输出pm个feature map，0<p<=1为压缩因子，当p小于1时，为DenseNet-C，同时使用bottleneck layer为DenseNet-BC

class DenseNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers, dropout_prob, class_dim=5):
        super(DenseNet, self).__init__(name_scope)

        self.layers = layers
        self.dropout_prob = dropout_prob

        # DenseNet的配置。分别为DenseNet-121 DenseNet-169 DenseNet-201 DenseNet-161
        # 使用哪种网络由输入的参数决定(layers)
        # 121 表示Dense Block 以及 Translation Layer中一共有多少层，包括之前的卷积、池化，忽略最后的全连接层
        # （6+12+24+16）*2+3(Transition Layer)+2(Conv +pool) = 121
        # k=32表示growth Rate，每层DenseBlock输出的FeatureMap厚度
        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        # DenseNet 第一层卷积 大小为7*7的卷积核，步长为2
        self.conv1 = Conv2D(
            self.full_name(),
            num_filters=layer_conf[0] * 2,  # 使用2k个filter
            filter_size=7,  # 7*7卷积
            stride=2,  # 步长为2，每进行一次卷积后向后移动两个单位
            padding=3,  # 在周围补齐宽度为3的0，使得可以对每一个位置进行卷积
            groups=1,
            act=None,
            bias_attr=False)

        # DenstNet 第二层池化，使用最大池化，输出channel为上一步中得到的结果
        # 使用了2k个filter，因此channels = 2k
        self.pool1 = Pool2D(
            self.full_name(),
            pool_size=3,
            pool_padding=1,
            pool_stride=2,
            pool_type='max')
        channels = layer_conf[0] * 2

        self.convs = []
        # layer_conf中保存的list的长度代表了一共有几个dense block，
        # 由于最后一个denseblock后没有translation layer，此处只循环加入len-1个，最后一个dense Block单独添加
        # D T D T D T D
        for i in range(len(layer_conf[1]) - 1):
            # LoopLayer中包含了一个dense block 和一个 translation layer
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                LoopLayer(
                    self.full_name(),
                    num_filters=layer_conf[0],
                    num_channels=channels,
                    block_num=layer_conf[1][i],
                    drop_out_prob=self.dropout_prob
                ))
            channels = layer_conf[0]
            self.convs.append(conv_block)

            # 由于最后一个Dense block后不再需要translation layer进行过渡（降维）,因此单独添加
        self.conv3 = DenseBlock(
            self.full_name(),
            num_filters=layer_conf[1][-1],
            num_channels=layer_conf[0],
            block_num=layer_conf[0],
            drop_out_prob=self.dropout_prob)

        # 全连接层之前的全局平均池化
        self.pool2 = Pool2D(
            self.full_name(),
            global_pooling=True,
            pool_type='avg')

        # 全连接层，激活函数为softmax
        self.fc = FC(self.full_name(),
                     size=class_dim,
                     act='softmax')

    def forward(self, inputs, label=None):
        y = self.conv1(inputs)
        y = self.pool1(y)
        #   print(len(self.convs))
        for conv in self.convs:
            y = conv(y)
        y = self.conv3(y)
        y = self.pool2(y)
        y = self.fc(y)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y

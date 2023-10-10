import os
import random
import json
import zipfile
import numpy as np
import PIL.Image as Image
from paddle.io import Dataset
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
import matplotlib.pyplot as plt


def _make_divisible(v, divisor, min_value=None):
    """
    判断v能否整除divisor
    params:
            v: 被除数
            divisior: 除数
    return：如果v不能整除divisor,则往上取一个能整除divisor的数返回
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 act=None):
        """
        params:
                num_channels, 卷积层的输入通道数
                num_filters, 卷积层的输出通道数
                filter_size, 卷积核的大小
                stride, 卷积层的步长
                padding, 填充大小，默认 padding=0 不填充
                groups, 分组卷积的组数，默认groups=1不使用分组卷积
                act, 激活函数类型，默认 act=None 不使用激活函数
        """
        super(ConvBNReLU, self).__init__()
        # 创建卷积层
        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups)
        # 创建 BatchNorm 层
        self.batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class InvertedResidual(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        """
        params:
                expand_ratio: 1x1升维通道数的扩张比例
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, filter_size=1))  # 1x1卷积升维
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),  # 深度可分离卷积
            Conv2D(hidden_dim, out_channels, filter_size=1, stride=1, padding=0),  # 1x1卷积降维
            BatchNorm(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, num_classes, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t=expand_ratio: 1x1升维扩张的倍数
                # c：输出通道数
                # n：次数
                # s：步长
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, filter_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

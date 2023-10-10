from functools import partial

import mindspore.nn as nn
import mindspore.ops as ops

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.pooling import GlobalAvgPooling


class Inception(nn.Cell):
    """
    Inception Block
    """

    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        norm = partial(nn.BatchNorm2d, eps=0.001)
        self.b1 = Conv2dNormActivation(in_channels, n1x1, kernel_size=1, norm=norm)
        self.b2 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, n3x3red, kernel_size=1, norm=norm),
            Conv2dNormActivation(n3x3red, n3x3, kernel_size=3, pad_mode='same', norm=norm)
        ])
        self.b3 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, n5x5red, kernel_size=1, norm=norm),
            Conv2dNormActivation(n5x5red, n5x5, kernel_size=3, pad_mode='same', norm=norm)
        ])

        self.b4 = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same'),
            Conv2dNormActivation(in_channels, pool_planes, kernel_size=1, norm=norm)
        ])

        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        branch4 = self.b4(x)
        return self.concat((branch1, branch2, branch3, branch4))


class GoogleNet(nn.Cell):
    """
    Googlenet architecture
    """

    # 添加了drop_out参数，去掉了include参数
    def __init__(self, num_classes, dropout=0.2):
        super(GoogleNet, self).__init__()
        norm = partial(nn.BatchNorm2d, eps=0.001)
        self.conv1 = Conv2dNormActivation(3, 64, kernel_size=7, stride=2, pad_mode='same', norm=norm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.conv2 = Conv2dNormActivation(64, 64, kernel_size=1, norm=norm)
        self.conv3 = Conv2dNormActivation(64, 192, kernel_size=3, pad_mode='same', norm=norm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.block3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.block3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.block4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.block4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.block4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.block4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.block4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        self.block5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.block5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.pool = GlobalAvgPooling()  # 这里使用公用的GlobalAvgPooling做全局部平均池化

        self.classifier = nn.Dense(1024, num_classes)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.block3a(x)
        x = self.block3b(x)
        x = self.maxpool3(x)

        x = self.block4a(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)
        x = self.block4e(x)
        x = self.maxpool4(x)

        x = self.block5a(x)
        x = self.block5b(x)

        x = self.pool(x)
        x = self.classifier(x)

        return x


def googlenet(num_classes: int = 10, num_channel: int = 1, pretrained: bool = False) -> GoogLeNet:
    model = GoogLeNet(num_classes=num_classes, in_channels=num_channel)

    return model

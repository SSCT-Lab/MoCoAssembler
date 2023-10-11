import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class single_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention_block(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2D(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttU_Net(nn.Layer):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2D(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool1(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool2(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool3(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = paddle.concat(x=[x4, d5], axis=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = paddle.concat(x=[x3, d4], axis=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = paddle.concat(x=[x2, d3], axis=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = paddle.concat(x=[x1, d2], axis=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

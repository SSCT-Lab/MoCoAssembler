%matplotlib inline
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.datasets import Cifar10, Cifar100
from paddle.vision.transforms import Transpose
from paddle.io import Dataset, DataLoader
from paddle import nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import paddlex
from paddle import ParamAttr

class BA_module(nn.Layer):
    def __init__(self, pre_channels, cur_channel, reduction=16):
        super().__init__()
        self.pre_fusions = nn.LayerList(
            [nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(pre_channel, cur_channel // reduction, 1, bias_attr=False),
                nn.BatchNorm2D(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(cur_channel, cur_channel // reduction, 1, bias_attr=False),
                nn.BatchNorm2D(cur_channel // reduction)
            )

        self.generation = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(cur_channel // reduction, cur_channel, 1, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.shape

        pre_fusions = [self.pre_fusions[i](pre_layers[i]) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer)
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion)

        return att_weights
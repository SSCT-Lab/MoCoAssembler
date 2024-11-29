import os
import jittor
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import copy

jittor.cudnn.set_max_workspace_ratio(0.0)


class alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_mutated = jittor.nn.AdaptiveAvgPool2d(output_size=8)
        self.relu1_mutated = jittor.nn.ReLU()
        self.pool1_mutated = jittor.nn.PixelShuffle(upscale_factor=8)

    def execute(self, x):
        x = self.conv1_mutated(x)
        x = self.relu1_mutated(x)
        x = self.pool1_mutated(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = alexnet()
    y = m(x)
    return list(y.shape)

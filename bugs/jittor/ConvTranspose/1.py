import os

os.environ["disable_lock"] = "1"
import jittor
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import copy


class alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.relu1 = jittor.nn.ReLU()
        self.pool1 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = jittor.nn.ConvTranspose(in_channels=64, kernel_size=5, out_channels=192, output_padding=(2,))

    def execute(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = alexnet()
    y = m(x)
    return list(y.shape)

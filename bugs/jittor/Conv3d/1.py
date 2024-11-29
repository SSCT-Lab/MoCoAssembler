import os

os.environ["disable_lock"] = "1"
import jittor
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import copy


class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.Conv(in_channels=1, out_channels=6, kernel_size=5)
        self.relu1 = jittor.nn.ReLU()
        self.pool1 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = jittor.nn.Conv3d(in_channels=6, kernel_size=5, out_channels=16, dilation=(2, 7, 0))

    def execute(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 1, 28, 28])
    m = lenet()
    y = m(x)
    return list(y.shape)


go()
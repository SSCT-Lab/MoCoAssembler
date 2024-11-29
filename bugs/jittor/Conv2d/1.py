import os

os.environ["disable_lock"] = "1"
import jittor
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import copy


class googlenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_mutated = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=5, padding=3,
                                              bias=False)
        self.relu1_mutated = jittor.nn.ReLU6()
        self.maxpool1_mutated = jittor.nn.ELU()
        self.conv2_mutated = jittor.nn.ConvTranspose(in_channels=64, kernel_size=1, out_channels=64)
        self.relu2_mutated = jittor.nn.ReLU()
        self.conv3_mutated = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(6, 5), stride=1, padding=1,
                                              bias=True, groups=8, dilation=1)
        self.relu3_mutated = jittor.nn.PReLU()
        self.maxpool2_mutated = jittor.nn.ELU()
        self.inception3a = Inception8766066066580()

    def execute(self, x):
        x = self.conv1_mutated(x)
        x = self.relu1_mutated(x)
        x = self.maxpool1_mutated(x)
        x = self.conv2_mutated(x)
        x = self.relu2_mutated(x)
        x = self.conv3_mutated(x)
        x = self.relu3_mutated(x)
        x = self.maxpool2_mutated(x)
        x = self.inception3a(x)
        return x


class Inception8766066066580(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_mutated = jittor.nn.Softplus()
        self.relu1_mutated = jittor.nn.ReLU()
        self.conv2a_mutated = jittor.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=2, stride=3, groups=7,
                                               bias=False, padding=(8, 3), dilation=(3, 8))
        self.relu2a_mutated = jittor.nn.ReLU()
        self.conv2b_mutated = jittor.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=8, stride=(5, 3),
                                               padding=8, bias=False, dilation=8, groups=4)
        self.relu2b_mutated = jittor.nn.ReLU()
        self.conv3a_mutated = jittor.nn.Conv2d(in_channels=192, out_channels=16, kernel_size=8, stride=(6, 5),
                                               dilation=(8, 1), padding=(5, 2), groups=8, bias=False)
        self.relu3a_mutated = jittor.nn.ReLU()
        self.conv3b_mutated = jittor.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=5, padding=(6, 4),
                                               bias=False, dilation=1)
        self.relu3b_mutated = jittor.nn.ReLU()
        self.conv4_mutated = jittor.nn.Conv2d(in_channels=192, out_channels=32, kernel_size=4, stride=(8, 1), groups=1,
                                              bias=True, dilation=(1, 2), padding=(8, 5))
        self.pool_mutated = jittor.nn.MaxPool2d(kernel_size=(8, 4), stride=8, padding=5, return_indices=True,
                                                ceil_mode=True)
        self.relu4_mutated = jittor.nn.ReLU()
        self.cat = jittor.concat

    def execute(self, x):
        branch1 = self.conv1_mutated(x)
        branch1 = self.relu1_mutated(branch1)
        branch2 = self.conv2a_mutated(x)
        branch2 = self.relu2a_mutated(branch2)
        branch2 = self.conv2b_mutated(branch2)
        branch2 = self.relu2b_mutated(branch2)
        branch3 = self.conv3a_mutated(x)
        branch3 = self.relu3a_mutated(branch3)
        branch3 = self.conv3b_mutated(branch3)
        branch3 = self.relu3b_mutated(branch3)
        branch4 = self.conv4_mutated(x)
        branch4 = self.pool_mutated(branch4)
        branch4 = self.relu4_mutated(branch4)
        x = self.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = googlenet()
    y = m(x)
    return list(y.shape)

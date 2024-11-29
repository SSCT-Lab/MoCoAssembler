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
        self.conv1_mutated = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=6, stride=4, padding=2)
        self.relu1_mutated = jittor.nn.ReLU()
        self.pool1_mutated = jittor.nn.ZeroPad2d(padding=(1, 5, 7, 1))
        self.conv2_mutated = jittor.nn.UpsamplingNearest2d(scale_factor=1.0)
        self.relu2_mutated = jittor.nn.GELU()
        self.pool2_mutated = jittor.nn.MaxPool2d(kernel_size=(3, 6), stride=8, ceil_mode=False, return_indices=False,
                                                 padding=(6, 3))
        self.conv3_mutated = jittor.nn.Sigmoid()
        self.relu3_mutated = jittor.nn.ELU()
        self.conv4_mutated = jittor.nn.MaxPool2d(kernel_size=(3, 7), return_indices=False, stride=(6, 2))
        self.relu4_mutated = jittor.nn.ReLU()
        self.conv5_mutated = jittor.nn.Flatten()
        self.relu5_mutated = jittor.nn.Softmax()
        self.pool3_mutated = jittor.nn.ConstantPad2d(padding=5, value=0.0)

    def execute(self, x):
        x = self.conv1_mutated(x)
        x = self.relu1_mutated(x)
        x = self.pool1_mutated(x)
        x = self.conv2_mutated(x)
        x = self.relu2_mutated(x)
        x = self.pool2_mutated(x)
        x = self.conv3_mutated(x)
        x = self.relu3_mutated(x)
        x = self.conv4_mutated(x)
        x = self.relu4_mutated(x)
        x = self.conv5_mutated(x)
        x = self.relu5_mutated(x)
        print(x.shape)
        x = self.pool3_mutated(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = alexnet()
    y = m(x)
    return list(y.shape)


go()
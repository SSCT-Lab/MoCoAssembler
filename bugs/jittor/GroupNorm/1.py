import os

os.environ["disable_lock"] = "1"
import jittor
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import copy


class pointnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_mutated = jittor.nn.Conv1d(in_channels=3, kernel_size=1, out_channels=64)
        self.layer2_mutated = jittor.nn.Sigmoid()
        self.layer3_mutated = jittor.nn.ELU()
        self.layer4_mutated = jittor.nn.Tanh()
        self.layer5_mutated = jittor.nn.Sigmoid()
        self.layer6_mutated = jittor.nn.GELU()
        self.layer7_mutated = jittor.nn.Tanh()
        self.layer8_mutated = jittor.nn.GroupNorm(num_channels=1, num_groups=8)

    def execute(self, x):
        x = self.layer1_mutated(x)
        x = self.layer2_mutated(x)
        x = self.layer3_mutated(x)
        x = self.layer4_mutated(x)
        x = self.layer5_mutated(x)
        x = self.layer6_mutated(x)
        x = self.layer7_mutated(x)
        x = self.layer8_mutated(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([2, 3, 2048])
    m = pointnet()
    y = m(x)
    return list(y.shape)

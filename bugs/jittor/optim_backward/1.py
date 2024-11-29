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
        self.conv1_mutated = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 1), stride=2, padding=3,
                                              dilation=(2, 8))
        self.relu1_mutated = jittor.nn.ELU()
        self.maxpool1_mutated = jittor.nn.MaxPool2d(kernel_size=(4, 6), stride=8, ceil_mode=False, return_indices=False,
                                                    padding=8)
        self.conv2_mutated = jittor.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=(1, 8),
                                              padding=(3, 2), dilation=(4, 3), bias=True, groups=8)
        self.relu2_mutated = jittor.nn.ReLU()
        self.conv3_mutated = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(1, 5), stride=8, padding=7,
                                              groups=1, bias=True, dilation=2)
        self.relu3_mutated = jittor.nn.ReLU6()
        self.maxpool2_mutated = jittor.nn.Sigmoid()
        self.inception3a = Inception8728474281792()
        self.inception3b = Inception8728474288650()
        self.maxpool3_mutated = jittor.nn.Sigmoid()
        self.inception4a = Inception8728474288842()
        self.tail_flatten = jittor.nn.Flatten()
        self.tail_fc = jittor.nn.Linear(in_features=5120, out_features=1000)

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
        x = self.inception3b(x)
        x = self.maxpool3_mutated(x)
        x = self.inception4a(x)
        x = self.tail_flatten(x)
        x = self.tail_fc(x)
        return x


class Inception8728474281792(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.relu1 = jittor.nn.ReLU()
        self.conv2a = jittor.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1)
        self.relu2a = jittor.nn.ReLU()
        self.conv2b = jittor.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2b = jittor.nn.ReLU()
        self.conv3a = jittor.nn.Conv2d(in_channels=192, out_channels=16, kernel_size=1, stride=1)
        self.relu3a_mutated = jittor.nn.ReLU()
        self.conv3b = jittor.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3b = jittor.nn.ReLU()
        self.conv4 = jittor.nn.Conv2d(in_channels=192, out_channels=32, kernel_size=1, stride=1)
        self.pool = jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu4 = jittor.nn.ReLU()
        self.cat = jittor.concat

    def execute(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)
        branch2 = self.conv2a(x)
        branch2 = self.relu2a(branch2)
        branch2 = self.conv2b(branch2)
        branch2 = self.relu2b(branch2)
        branch3 = self.conv3a(x)
        branch3 = self.relu3a_mutated(branch3)
        branch3 = self.conv3b(branch3)
        branch3 = self.relu3b(branch3)
        branch4 = self.conv4(x)
        branch4 = self.pool(branch4)
        branch4 = self.relu4(branch4)
        x = self.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


class Inception8728474288650(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.relu1 = jittor.nn.ReLU()
        self.conv2a_mutated = jittor.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.relu2a = jittor.nn.ReLU()
        self.conv2b = jittor.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.relu2b = jittor.nn.ReLU()
        self.conv3a = jittor.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1)
        self.relu3a_mutated = jittor.nn.ReLU()
        self.conv3b = jittor.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=5, stride=1, padding=2)
        self.relu3b = jittor.nn.ReLU()
        self.conv4 = jittor.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.pool = jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu4 = jittor.nn.ReLU()
        self.cat = jittor.concat

    def execute(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)
        branch2 = self.conv2a_mutated(x)
        branch2 = self.relu2a(branch2)
        branch2 = self.conv2b(branch2)
        branch2 = self.relu2b(branch2)
        branch3 = self.conv3a(x)
        branch3 = self.relu3a_mutated(branch3)
        branch3 = self.conv3b(branch3)
        branch3 = self.relu3b(branch3)
        branch4 = self.conv4(x)
        branch4 = self.pool(branch4)
        branch4 = self.relu4(branch4)
        x = self.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


class Inception8728474288842(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=480, out_channels=192, kernel_size=1)
        self.relu1 = jittor.nn.ReLU()
        self.conv2a = jittor.nn.Conv2d(in_channels=480, out_channels=96, kernel_size=1, stride=1)
        self.relu2a = jittor.nn.ReLU()
        self.conv2b_mutated = jittor.nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, stride=(7, 2),
                                               padding=1)
        self.relu2b = jittor.nn.ReLU()
        self.conv3a = jittor.nn.Conv2d(in_channels=480, out_channels=16, kernel_size=1, stride=1)
        self.relu3a = jittor.nn.ReLU()
        self.conv3b = jittor.nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.relu3b = jittor.nn.ReLU()
        self.conv4 = jittor.nn.Conv2d(in_channels=480, out_channels=64, kernel_size=1, stride=1)
        self.pool = jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu4 = jittor.nn.ReLU()
        self.cat = jittor.concat

    def execute(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)
        branch2 = self.conv2a(x)
        branch2 = self.relu2a(branch2)
        branch2 = self.conv2b_mutated(branch2)
        branch2 = self.relu2b(branch2)
        branch3 = self.conv3a(x)
        branch3 = self.relu3a(branch3)
        branch3 = self.conv3b(branch3)
        branch3 = self.relu3b(branch3)
        branch4 = self.conv4(x)
        branch4 = self.pool(branch4)
        branch4 = self.relu4(branch4)
        x = self.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = googlenet()
    y = m(x)
    return list(y.shape)


def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 9999999
    else:
        return float(np.max(np.abs(A - B)))


def train(x, x_t, y_t):
    flag = True
    jittor.flags.use_cuda = 0
    m_c = googlenet()
    opt_c = optim.SGD(m_c.parameters(), lr=0.01)

    jittor.flags.use_cuda = 1
    m_g = copy.deepcopy(m_c)
    opt_g = optim.SGD(m_g.parameters(), lr=0.01)

    jittor.flags.use_cuda = 0
    input_c = jittor.array(x_t).float32()
    input_c = input_c.ceil_int()
    target_c = jittor.array(y_t)
    output_c = m_c(input_c)
    loss_c = nn.CrossEntropyLoss()(output_c, target_c)
    opt_c.backward(loss_c)

    jittor.flags.use_cuda = 1
    input_g = jittor.array(x_t).float32()
    input_g = input_g.ceil_int()
    target_g = jittor.array(y_t)
    output_g = m_g(input_g)
    loss_g = nn.CrossEntropyLoss()(output_g, target_g)
    opt_g.backward(loss_g)

    output_c_np = output_c.fetch_sync()
    output_g_np = output_g.fetch_sync()

    jittor.flags.use_cuda = 0
    if chebyshev_distance(output_c_np, output_g_np) > 0.1:
        flag = False
        jittor.clean()
        return flag, 'Output diff too big'
    if abs(loss_c.item() - loss_g.item()) > 0.1:
        flag = False
        jittor.clean()
        return flag, 'Loss diff too big'
    for (param_c, param_g) in zip(m_c.parameters(), m_g.parameters()):
        weights_c = param_c
        weights_g = param_g
        distance = chebyshev_distance(weights_c, weights_g)
        if distance > 0.1:
            flag = False
            break
    if not flag:
        jittor.clean()
        return flag, 'Grad diff too big'

    jittor.clean()
    return flag, ''

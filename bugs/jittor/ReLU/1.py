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
        self.conv1_mutated = jittor.nn.ConvTranspose2d(in_channels=3, kernel_size=11, out_channels=64)
        self.relu1_mutated = jittor.nn.Softmax()
        self.pool1_mutated = jittor.nn.ReplicationPad2d(padding=8)
        self.conv2_mutated = jittor.nn.ELU()
        self.relu2_mutated = jittor.nn.LeakyReLU()
        self.pool2_mutated = jittor.nn.MaxPool2d(kernel_size=3, stride=8, return_indices=False, ceil_mode=True)
        self.conv3_mutated = jittor.nn.Conv2d(in_channels=64, out_channels=384, kernel_size=(4, 8), padding=(8, 3),
                                              stride=8, groups=2, bias=False, dilation=(1, 1))
        self.relu3_mutated = jittor.nn.ReLU()
        self.tail_flatten = jittor.nn.Flatten()
        self.tail_fc = jittor.nn.Linear(in_features=9216, out_features=1000)

    def execute(self, x):
        x = self.conv1_mutated(x)
        x = self.relu1_mutated(x)
        x = self.pool1_mutated(x)
        x = self.conv2_mutated(x)
        x = self.relu2_mutated(x)
        x = self.pool2_mutated(x)
        x = self.conv3_mutated(x)
        x = self.relu3_mutated(x)
        x = self.tail_flatten(x)
        x = self.tail_fc(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = alexnet()
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
    m_c = alexnet()
    opt_c = optim.SGD(m_c.parameters(), lr=0.01)

    jittor.flags.use_cuda = 1
    m_g = copy.deepcopy(m_c)
    opt_g = optim.SGD(m_g.parameters(), lr=0.01)

    jittor.flags.use_cuda = 0
    input_c = jittor.array(x_t).float32()
    input_c = jittor.ceil_int(input_c)
    target_c = jittor.array(y_t)
    output_c = m_c(input_c)
    loss_c = nn.CrossEntropyLoss()(output_c, target_c)
    opt_c.backward(loss_c)

    jittor.flags.use_cuda = 1
    input_g = jittor.array(x_t).float32()
    input_g = jittor.ceil_int(input_g)
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

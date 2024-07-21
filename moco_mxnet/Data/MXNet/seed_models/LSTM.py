import mxnet
import mxnet.gluon.nn as nn
import numpy as np
from mxnet import optim
import os
import mxnet.gluon.nn.functional as F


class Model_wOzBoAGFnY8gddTApgZ2fqdx5PVjsOSA(nn.Module):
    def __init__(self):
        super(Model_wOzBoAGFnY8gddTApgZ2fqdx5PVjsOSA, self).__init__()
        self.rnn1 = mxnet.gluon.nn.RNN(input_size=2048, hidden_size=1076, batch_first=True)
        self.rnn2 = mxnet.gluon.nn.RNN(input_size=1076, hidden_size=723, batch_first=True)
        self.rnn3 = mxnet.gluon.nn.LSTM(input_size=723, hidden_size=662, batch_first=True)
        self.rnn4 = mxnet.gluon.nn.RNN(input_size=662, hidden_size=398, batch_first=True)
        self.linear = mxnet.gluon.nn.Linear(in_features=398, out_features=10)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x, _ = self.rnn3(x)
        _, x = self.rnn4(x)
        x = self.linear(x[-1])

        x = x
        return x


def go():
    try:
        model = Model_wOzBoAGFnY8gddTApgZ2fqdx5PVjsOSA().to('cuda')
        x = mxnet.randn([1, 3, 2048]).to('cuda')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_wOzBoAGFnY8gddTApgZ2fqdx5PVjsOSA(nn.Module):
    def __init__(self):
        super(Model_wOzBoAGFnY8gddTApgZ2fqdx5PVjsOSA, self).__init__()
        self.rnn1 = torch.nn.RNN(input_size=2048, hidden_size=1076, batch_first=True)
        self.rnn2 = torch.nn.RNN(input_size=1076, hidden_size=723, batch_first=True)
        self.rnn3 = torch.nn.LSTM(input_size=723, hidden_size=662, batch_first=True)
        self.rnn4 = torch.nn.RNN(input_size=662, hidden_size=398, batch_first=True)
        self.linear = torch.nn.Linear(in_features=398, out_features=10)

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
        x = torch.randn([1, 3, 2048]).to('cuda')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag

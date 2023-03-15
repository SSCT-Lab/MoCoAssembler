# -*- coding: utf-8 -*-

"""
@Title   : Pytorch implementation of GRU
@Time    : Mar. 13th, 2023
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""

from torch import nn
import torch


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()

        # reset gate
        self.rx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.rh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # update gate
        self.zx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.zh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # tanh
        self.hx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.hh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x, h_1):
        r = self.rx_linear(x) + self.rh_linear(h_1)
        r = torch.sigmoid(r)

        z = self.zx_linear(x) + self.zh_linear(h_1)
        z = torch.sigmoid(z)

        h_ = self.hx_linear(x) + self.hh_linear(r * h_1)
        h_ = torch.tanh(h_)

        output = z * h_1 + (1 - z) * h_
        return output


class GRU(nn.Module):

    def __init__( self, input_dim, hidden_dim):
        super(GRU, self).__init__( )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gruCell = GRUCell(input_dim, hidden_dim)

    def forward(self, x):
        outputs = []
        layer = None
        for seq_lens in x:
            if layer is None:
                layer = torch.randn(x.shape[1], self.hidden_dim)

            layer = self.gruCell(seq_lens, layer)
            outputs.append(torch.unsqueeze(layer, 0))

        outputs = torch.cat(outputs)
        return outputs, layer


if __name__ == '__main__':
    x = torch.randn(24, 12)
    h = torch.randn(24, 6)
    rc = GRUCell(12, 6)
    h = rc(x, h)
    print(h.shape)

    net = GRU(12, 6)
    x = torch.randn(5, 24, 12)
    outputs, h = net(x)
    print(outputs.shape)
    print(h.shape)

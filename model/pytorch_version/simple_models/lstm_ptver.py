import torch
from torch import Tensor
from torch.nn import Parameter, init, Module

import math


class LSTM(Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size  # 输入的大小，一般
        self.hidden_size = hidden_size  # 隐藏层输出大小，也就是单元个数

        # input gate
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x, state):
        global h_next_t, c_next_t
        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()
        hidden_seq = []
        seq_size = 1
        for t in range(seq_size):
            p = x[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii @ p + self.b_ii + self.w_hi @ h_t + self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ p + self.b_if + self.w_hf @ h_t + self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ p + self.b_ig + self.w_hg @ h_t + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ p + self.b_io + self.w_ho @ h_t + self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        x = torch.cat(hidden_seq, dim=0)
        return x


if __name__ == '__main__':
    inputs, h0, c0 = torch.ones(1, 1, 10), torch.ones(1, 1, 20), torch.ones(1, 1, 20)
    lstm = LSTM(10, 20)
    output1 = lstm(inputs, (h0, c0))
    print(output1.shape)

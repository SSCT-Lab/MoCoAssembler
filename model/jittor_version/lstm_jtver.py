import jittor as jt
import jittor.nn as nn
from jittor import Var
from jittor.nn import init, Module
import math


class LSTM(Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size  # 输入的大小，一般
        self.hidden_size = hidden_size  # 隐藏层输出大小，也就是单元个数

        # 这里没有torch中那种自动用随机小值来初始化张量的Parameter，手动进行

        # input gate
        self.w_ii = jt.randn(hidden_size, input_size) * 1e-40
        self.w_hi = jt.randn(hidden_size, hidden_size) * 1e-40
        self.b_ii = jt.randn(hidden_size, 1) * 1e-40
        self.b_hi = jt.randn(hidden_size, 1) * 1e-40

        # forget gate
        self.w_if = jt.randn(hidden_size, input_size) * 1e-40
        self.w_hf = jt.randn(hidden_size, hidden_size) * 1e-40
        self.b_if = jt.randn(hidden_size, 1) * 1e-40
        self.b_hf = jt.randn(hidden_size, 1) * 1e-40

        # output gate
        self.w_io = jt.randn(hidden_size, input_size) * 1e-40
        self.w_ho = jt.randn(hidden_size, hidden_size) * 1e-40
        self.b_io = jt.randn(hidden_size, 1) * 1e-40
        self.b_ho = jt.randn(hidden_size, 1) * 1e-40

        # cell
        self.w_ig = jt.randn(hidden_size, input_size) * 1e-40
        self.w_hg = jt.randn(hidden_size, hidden_size) * 1e-40
        self.b_ig = jt.randn(hidden_size, 1) * 1e-40
        self.b_hg = jt.randn(hidden_size, 1) * 1e-40

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def execute(self, x, state):
        global h_next_t, c_next_t
        if state is None:
            h_t = jt.zeros(1, self.hidden_size).t()
            c_t = jt.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()
        hidden_seq = []
        seq_size = 1
        for t in range(seq_size):
            p = x[:, t, :].t()
            # input gate
            i = jt.sigmoid(self.w_ii @ p + self.b_ii + self.w_hi @ h_t + self.b_hi)
            # forget gate
            f = jt.sigmoid(self.w_if @ p + self.b_if + self.w_hf @ h_t + self.b_hf)
            # cell
            g = jt.tanh(self.w_ig @ p + self.b_ig + self.w_hg @ h_t + self.b_hg)
            # output gate
            o = jt.sigmoid(self.w_io @ p + self.b_io + self.w_ho @ h_t + self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * jt.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        x = jt.cat(hidden_seq, dim=0)
        return x


if __name__ == '__main__':
    inputs, h0, c0 = jt.ones((1, 1, 10)), jt.ones((1, 1, 20)), jt.ones((1, 1, 20))
    lstm = LSTM(10, 20)
    output1 = lstm(inputs, (h0, c0))
    print(output1.shape)

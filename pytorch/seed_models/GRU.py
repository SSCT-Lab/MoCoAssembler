import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size=20, hidden_size=100, output_size=20):
        super(GRU, self).__init__()

        self.gru_cell = torch.nn.GRUCell(input_size=1, hidden_size=1)
        self.fc = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        h0 = torch.zeros((batch_size, seq_length))

        outputs = []
        hn = h0
        func = self.gru_cell
        for t in range(seq_length):
            hn = func(x[:, t, :], hn)
            outputs.append(hn)

        x = self.fc(outputs[-1])

        return x


def go():
    net = GRU(1, 1, 1)
    x = torch.randn((5, 1, 1))
    y = net(x)
    return net

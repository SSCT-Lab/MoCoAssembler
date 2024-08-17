import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.hidden_size = 200
        self.num_layers = 4

        self.lstm1 = torch.nn.LSTMCell(input_size=2048, hidden_size=200)
        self.lstm2 = torch.nn.LSTMCell(input_size=200, hidden_size=200)
        self.lstm3 = torch.nn.LSTMCell(input_size=200, hidden_size=200)
        self.lstm4 = torch.nn.LSTMCell(input_size=200, hidden_size=200)

        self.fc = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        if True:
            h0 = torch.zeros((batch_size, self.hidden_size))
            c0 = torch.zeros((batch_size, self.hidden_size))

        hn1, cn1 = h0, c0
        hn2, cn2 = h0, c0
        hn3, cn3 = h0, c0
        hn4, cn4 = h0, c0

        for t in range(seq_length):
            hn0 = x[:, t, :]
            hn1, cn1 = self.lstm1(hn0, (hn1, cn1))
            hn2, cn2 = self.lstm2(hn1, (hn2, cn2))
            hn3, cn3 = self.lstm3(hn2, (hn3, cn3))
            hn4, cn4 = self.lstm4(hn3, (hn4, cn4))

        x = self.fc(hn4)

        return x


def go():
    x = torch.randn(1, 3, 2048)
    net = LSTM()
    y = net(x)
    print(y.shape)
    return net

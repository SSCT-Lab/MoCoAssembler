import jittor
import jittor.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=200, num_layers=4, output_size=20):
        super(LSTM, self).__init__()

        self.hidden_size = 200
        self.num_layers = 4

        self.lstm1 = jittor.nn.LSTMCell(input_size=1, hidden_size=200)
        self.lstm2 = jittor.nn.LSTMCell(input_size=200, hidden_size=200)
        self.lstm3 = jittor.nn.LSTMCell(input_size=200, hidden_size=200)
        self.lstm4 = jittor.nn.LSTMCell(input_size=200, hidden_size=200)

        self.fc = nn.Linear(in_features=200, out_features=1)

    def execute(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        if True:
            h0 = jittor.zeros((batch_size, self.hidden_size))
            c0 = jittor.zeros((batch_size, self.hidden_size))

        hn1, cn1 = h0, c0
        hn2, cn2 = h0, c0
        hn3, cn3 = h0, c0
        hn4, cn4 = h0, c0

        f1 = self.lstm1
        f2 = self.lstm2
        f3 = self.lstm3
        f4 = self.lstm4
        for t in range(seq_length):
            hn1, cn1 = f1(x[:, t, :], (hn1, cn1))
            hn2, cn2 = f2(hn1, (hn2, cn2))
            hn3, cn3 = f3(hn2, (hn3, cn3))
            hn4, cn4 = f4(hn3, (hn4, cn4))

        x = self.fc(hn4)

        return x


def go():
    HIDDEN_SIZE = 200
    NUM_LAYERS = 4
    OUTPUT_SIZE = 20
    x = jittor.randn(5, 1, 1)
    net = LSTM()
    y = net(x)
    return net

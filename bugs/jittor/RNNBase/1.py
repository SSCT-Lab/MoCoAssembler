import jittor
import jittor.nn as nn


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.layer1 = jittor.nn.LSTMCell(hidden_size=512, input_size=1)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.RNNBase(hidden_size=200, input_size=200, mode="LSTM")

    def execute(self, x):
        x, _ = self.layer1(x)
        x = self.layer2(x)
        x, _ = self.layer3(x)
        return x


def go():
    model = lstm()
    x = jittor.randn(5, 1)
    y = model(x)
    print(y)
    return model

import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.layer1 = torch.nn.LSTMCell(hidden_size=200, input_size=1, bias=False)
        self.layer2 = torch.nn.Hardsigmoid(inplace=False)
        self.layer3 = torch.nn.LSTMCell(hidden_size=200, input_size=200, bias=True)
        self.layer4 = torch.nn.Hardswish(inplace=True)
        self.layer5 = torch.nn.LSTMCell(hidden_size=200, input_size=200, bias=False)
        self.layer6 = torch.nn.SELU()
        self.layer7 = torch.nn.LSTM(hidden_size=200, input_size=200, proj_size=462)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = self.layer2(x)
        x, _ = self.layer3(x)
        x = self.layer4(x)
        x, _ = self.layer5(x)
        x = self.layer6(x)
        x, _ = self.layer7(x)
        return x


def go():
    model = lstm()
    x = torch.randn(5, 1)
    y = model(x)
    return model

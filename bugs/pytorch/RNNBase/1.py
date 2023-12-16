import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.layer1 = torch.nn.LSTMCell(hidden_size=200, input_size=1, bias=False)
        self.layer2 = torch.nn.ReLU6(inplace=True)
        self.layer3 = torch.nn.LSTMCell(hidden_size=200, input_size=200, bias=False)
        self.layer4 = torch.nn.ELU(alpha=0.9017482681196566, inplace=True)
        self.layer5 = torch.nn.LSTM(hidden_size=True, input_size=200, bias=False)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = self.layer2(x)
        x, _ = self.layer3(x)
        x = self.layer4(x)
        x, _ = self.layer5(x)
        return x


def go():
    model = lstm()
    x = torch.randn(5, 1)
    y = model(x)
    return model.to("cuda")

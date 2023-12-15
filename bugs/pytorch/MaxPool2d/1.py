import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, kernel_size=5, out_channels=6, padding=8)
        self.layer2 = torch.nn.SELU(inplace=False)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=(1, 4), padding=7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def go():
    model = LeNet()
    x = torch.randn(3, 1, 28, 28)
    y = model(x)
    return model

import torch
import torch.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        return x


def go():
    device = torch.device('cuda')
    net = testnet().to(device)
    x = torch.randn(3, 3, 224, 224).to(device)
    y = net(x)

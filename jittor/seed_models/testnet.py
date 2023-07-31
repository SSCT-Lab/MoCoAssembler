import jittor
import jittor.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.relu1 = jittor.nn.ReLU()
        self.conv2 = jittor.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, padding=0)
        self.relu2 = jittor.nn.Sigmoid()
        self.pool = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

    def execute(self, x):
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


def go():
    net = testnet()
    x = jittor.randn(1, 4, 32, 32)
    y = net(x)
    return net

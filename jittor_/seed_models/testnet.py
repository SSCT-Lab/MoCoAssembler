import jittor
import jittor.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = jittor.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3)
        self.relu1 = jittor.nn.ReLU()
        self.relu2 = jittor.nn.ReLU()

    def execute(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


if __name__ == '__main__':
    net = testnet()
    x = jittor.randn(224, 3, 32, 32)
    y = net(x)
    print(y)

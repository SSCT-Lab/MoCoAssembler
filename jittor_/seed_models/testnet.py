import jittor
import jittor.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv1 = jittor.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.relu1 = jittor.nn.ReLU()
        self.conv2 = jittor.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)
        self.relu2 = jittor.nn.Sigmoid()
        self.pool = jittor.nn.MaxPool2d(2, 2)

    def execute(self, x):
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x


if __name__ == '__main__':
    net = testnet()
    x = jittor.randn(8, 4, 224, 224)
    y = net(x)
    print(y)

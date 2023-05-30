from torch import nn
from torchsummary import summary


class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.child = ChildA(3, 1)
        self.conv2a = nn.Conv2d(96, 16, kernel_size=1, stride=1)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.pool(x)
        x = self.child(x)
        x = self.conv2a(x)
        return x


class ChildA(nn.Module):
    def __init__(self, in_channels, ch1x1):
        super(ChildA, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.pool(x)
        return x


if __name__ == '__main__':
    net = DemoNet()
    summary(net, (3, 224, 224))

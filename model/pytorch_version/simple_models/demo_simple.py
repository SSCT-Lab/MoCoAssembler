from torch import nn
from torchsummary import summary


class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2a = nn.Conv2d(96, 16, kernel_size=1, stride=1)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2a(x)
        return x


if __name__ == '__main__':
    net = DemoNet()
    summary(net, (3, 224, 224))

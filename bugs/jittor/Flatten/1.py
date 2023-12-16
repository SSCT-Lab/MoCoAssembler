import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=1, kernel_size=5, out_channels=6, groups=1)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.layer4 = jittor.nn.Conv(in_channels=6, kernel_size=5, out_channels=16)
        self.layer5 = jittor.nn.ReLU()
        self.layer6 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, dilation=None)
        self.layer7 = jittor.nn.Flatten(end_dim=5)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


def go():
    model = LeNet()
    x = jittor.randn(3, 1, 28, 28)
    y = model(x)
    print(y)
    return model

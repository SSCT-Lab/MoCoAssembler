import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=1, kernel_size=5, out_channels=6, padding=(1, 8))
        self.layer2 = jittor.nn.Leaky_relu()
        self.layer3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer4 = jittor.nn.Conv2d(in_channels=6, kernel_size=5, out_channels=16, padding=0)
        self.layer5 = jittor.nn.ReLU()
        self.layer6 = jittor.nn.PixelShuffle(upscale_factor=-1)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


def go():
    model = LeNet()
    x = jittor.randn(3, 1, 28, 28)
    y = model(x)
    print(y)
    return model

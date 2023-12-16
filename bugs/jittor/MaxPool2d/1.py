import jittor
import jittor.nn as nn


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1)
        self.layer2 = jittor.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.MaxPool2d(kernel_size=3, stride=None, padding=-1)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def go():
    model = mobilenet()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

import jittor
import jittor.nn as nn


class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(4, 2))
        self.layer2 = jittor.nn.ReLU6()
        self.layer3 = jittor.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 6))
        self.layer4 = jittor.nn.LeakyReLU()
        self.layer5 = jittor.nn.AdaptiveMaxPool2d(output_size=196)
        self.layer6 = jittor.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1, 6))
        self.layer7 = jittor.nn.MaxPool2d(kernel_size=3, stride=-1)

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
    model = vgg19()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

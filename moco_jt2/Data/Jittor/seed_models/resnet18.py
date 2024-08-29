import jittor
import jittor.nn as nn


class resnet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(resnet18, self).__init__()
        self.conv = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = jittor.nn.BatchNorm2d(num_features=64)
        self.pool = jittor.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception1 = Inception(in_c=64, out_c=64, std=1)
        self.inception2 = Inception(in_c=64, out_c=64, std=1)
        self.inception3 = Inception(in_c=64, out_c=128, std=2)
        self.inception4 = Inception(in_c=128, out_c=128, std=1)
        self.inception5 = Inception(in_c=128, out_c=256, std=2)
        self.inception6 = Inception(in_c=256, out_c=256, std=1)
        self.inception7 = Inception(in_c=256, out_c=512, std=2)
        self.inception8 = Inception(in_c=512, out_c=512, std=1)
        self.pool2 = jittor.nn.AvgPool2d(kernel_size=7)
        self.flatten = jittor.nn.Flatten()
        self.linear = jittor.nn.Linear(in_features=512, out_features=1000)

    def execute(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_c, out_c, std):
        super().__init__()
        self.c1 = jittor.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=std, padding=1)
        self.b1 = jittor.nn.BatchNorm2d(num_features=out_c)
        self.r1 = jittor.nn.ReLU()
        self.c2 = jittor.nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.b2 = jittor.nn.BatchNorm2d(num_features=out_c)
        self.r2 = jittor.nn.ReLU()

    def execute(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        return x


def go():
    x = jittor.randn((1, 3, 224, 224))
    m = resnet18()
    y = m(x)
    return y.shape

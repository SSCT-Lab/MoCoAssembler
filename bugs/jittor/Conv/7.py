import jittor
import jittor.nn as nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.layer1 = jittor.nn.Conv(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.Pool(kernel_size=3, stride=2)
        self.layer4 = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2, stride=3)
        self.layer5 = jittor.nn.ReLU()
        self.layer6 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, return_indices=None)
        self.layer7 = jittor.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1, stride=1)
        self.layer8 = jittor.nn.ReLU6()
        self.layer9 = jittor.nn.Conv(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.layer10 = jittor.nn.ReLU()
        self.layer11 = jittor.nn.Conv2d(in_channels=256, out_channels=-1, kernel_size=3, padding=1, stride=(8, 4))

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x


def go():
    model = alexnet()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

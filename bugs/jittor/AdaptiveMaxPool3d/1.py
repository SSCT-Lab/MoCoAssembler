import jittor
import jittor.nn as nn


class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=3, kernel_size=7, out_channels=96, stride=2, bias=False)
        self.layer2 = jittor.nn.AdaptiveMaxPool2d(output_size=1)
        self.layer3 = jittor.nn.Conv2d(in_channels=96, kernel_size=1, out_channels=16, stride=4)
        self.layer4 = jittor.nn.GELU()
        self.layer5 = jittor.nn.ConvTranspose2d(in_channels=16, kernel_size=1, out_channels=64, stride=1)
        self.layer6 = jittor.nn.ReLU()
        self.layer7 = jittor.nn.Conv2d(in_channels=16, kernel_size=3, out_channels=64, padding=1, stride=1, bias=False)
        self.layer8 = jittor.nn.LeakyReLU()
        self.layer10 = jittor.nn.Tanh()
        self.layer11 = jittor.nn.Leaky_relu()
        self.layer12 = jittor.nn.Sigmoid()
        self.layer13 = jittor.nn.AdaptiveMaxPool3d(output_size=1)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y1 = self.layer5(x)
        y1 = self.layer6(y1)
        y2 = self.layer7(x)
        y2 = self.layer8(y2)
        x = jittor.cat([y1,y2], dim=-1)
        x = self.layer10(x)
        x = self.layer11(x)
        y1 = self.layer12(x)
        y1 = self.layer13(y1)
        return x


def go():
    model = squeezenet()
    x = jittor.randn(3, 3, 244, 244)
    y = model(x)
    print(y)
    return model

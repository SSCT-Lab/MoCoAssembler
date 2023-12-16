import jittor
import jittor.nn as nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.layer1 = jittor.nn.Conv(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, return_indices=None)
        self.layer4 = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=(1, 2))
        self.layer5 = jittor.nn.ReLU()
        self.layer6 = jittor.nn.AdaptiveMaxPool3d(output_size=33)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(jittor.unsqueeze(x, dim=0))
        return x


def go():
    model = alexnet()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

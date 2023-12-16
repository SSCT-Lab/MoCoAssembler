import jittor
import jittor.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.layer1 = jittor.nn.Conv1d(in_channels=3, kernel_size=1, out_channels=64, groups=1)
        self.layer2 = jittor.nn.BatchNorm1d(num_features=64, sync=True)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.Conv1d(in_channels=64, kernel_size=1, out_channels=128, dilation=(5, ))

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def go():
    model = pointnet()
    x = jittor.randn(3, 3, 5)
    y = model(x)
    print(y)
    return model

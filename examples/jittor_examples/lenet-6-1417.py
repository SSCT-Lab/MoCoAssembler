import copy

import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = jittor.nn.Conv(in_channels=1, out_channels=6, kernel_size=5, bias=False)
        self.conv_2 = jittor.nn.Sigmoid()
        self.conv_3 = jittor.nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv_4 = jittor.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv_5 = jittor.nn.Sigmoid()
        self.conv_6 = jittor.nn.MaxUnpool2d(kernel_size=2, stride=2)
    def execute(self, img):
        x = copy.deepcopy(img)
        # 1st block
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        # 2nd block
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        # 3rd block
        x = x.view(img.shape[0], -1)
        return x
def go():
    net = LeNet()
    y = net(jittor.randn((1, 1, 28, 28)))
    return net

import copy

import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = jittor.nn.Conv(in_channels=1, out_channels=6, kernel_size=5, bias=False)
        self.conv_2 = jittor.nn.Sigmoid()
        self.conv_3 = jittor.nn.Pool(kernel_size=2, stride=2)
        self.conv_4 = jittor.nn.Conv(in_channels=6, out_channels=16, kernel_size=5, padding=(5, 7))
        self.conv_5 = jittor.nn.Tanh()
    def execute(self, img):
        x = copy.deepcopy(img)
        # 1st block
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        # 2nd block
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x
def go():
    net = LeNet()
    y = net(jittor.randn((1, 1, 28, 28)))
    return net

import copy

import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = jittor.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv_2 = jittor.nn.Sigmoid()
        self.conv_3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.conv_4 = jittor.nn.Conv(in_channels=6, out_channels=16, kernel_size=(2, 4))
        self.conv_5 = jittor.nn.Tanh()
        self.conv_6 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, padding=6)
        self.fc_1 = jittor.nn.Tanh()
        self.fc_2 = jittor.nn.Sigmoid()
        self.fc_3 = jittor.nn.Linear(in_features=120, out_features=84, bias=True)
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
        x = self.fc_1(x)
        x = self.fc_2(x)

        # 4th block
        x = self.fc_3(x)
        return x
def go():
    net = LeNet()
    y = net(jittor.randn((1, 1, 28, 28)))
    return net

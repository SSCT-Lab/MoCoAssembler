import copy

import moco_jt
import moco_jt.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = moco_jt.nn.Conv(in_channels=1, out_channels=6, kernel_size=5)
        self.conv_2 = moco_jt.nn.Sigmoid()
        self.conv_3 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = moco_jt.nn.Conv(in_channels=6, out_channels=16, kernel_size=5)
        self.conv_5 = moco_jt.nn.Sigmoid()
        self.conv_6 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_1 = moco_jt.nn.Linear(in_features=256, out_features=120)
        self.fc_2 = moco_jt.nn.Sigmoid()

        self.fc_3 = moco_jt.nn.Linear(in_features=120, out_features=84)
        self.fc_4 = moco_jt.nn.Sigmoid()

        self.fc_5 = moco_jt.nn.Linear(in_features=84, out_features=10)

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
        x = self.fc_4(x)
        x = self.fc_5(x)

        return x


def go():
    net = LeNet()
    y = net(moco_jt.randn((1, 1, 28, 28)))
    return net

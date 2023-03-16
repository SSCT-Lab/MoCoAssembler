import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(2, 2), # kernel_size, stride
        #     nn.Conv2d(6, 16, 5),
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(2, 2)
        # )
        self.conv_1 = nn.Conv2d(1, 6, 5)
        self.conv_2 = nn.Sigmoid()
        self.conv_3 = nn.MaxPool2d(2, 2)

        self.conv_4 = nn.Conv2d(6, 16, 5)
        self.conv_5 = nn.Sigmoid()
        self.conv_6 = nn.MaxPool2d(2, 2)

        self.fc_1 = nn.Linear(16*4*4, 120)
        self.fc_2 = nn.Sigmoid()

        self.fc_3 = nn.Linear(120, 84)
        self.fc_4 = nn.Sigmoid()

        self.fc_5 = nn.Linear(84, 10)
        # self.fc = nn.Sequential(
        #     nn.Linear(16*4*4, 120),
        #     nn.Sigmoid(),
        #     nn.Linear(120, 84),
        #     nn.Sigmoid(),
        #     nn.Linear(84, 10)
        # )

    def forward(self, img):
        # feature = self.conv(img)
        # output = self.fc(feature.view(img.shape[0], -1))
        # 1st block
        x = self.conv_1(img)
        x = self.conv_2(x)
        x = self.conv_3(x)

        # 2nd block
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        #3rd block
        x = self.fc_1(x.view(img.shape[0], -1))
        x = self.fc_2(x)

        #4th block
        x = self.fc_3(x)
        x = self.fc_4(x)

        output = self.fc_5(x)

        return output


if __name__ == '__main__':
    net = LeNet()
    print(net)





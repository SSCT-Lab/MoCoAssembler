import copy

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv_2 = torch.nn.Sigmoid()
        self.conv_3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_4 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv_5 = torch.nn.Sigmoid()
        self.conv_6 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc_1 = torch.nn.SELU()
        self.fc_2 = torch.nn.Sigmoid()
        self.fc_3 = torch.nn.Bilinear(out_features=84, in1_features=1200, in2_features=1200)
    def forward(self, img):
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
    device = torch.device('cuda')
    net = LeNet().to(device)
    y = net(torch.randn((224, 1, 28, 28)).to(device))
    return net

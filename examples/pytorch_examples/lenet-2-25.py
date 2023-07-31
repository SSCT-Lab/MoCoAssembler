import copy

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv_2 = torch.nn.Sigmoid()
    def forward(self, img):
        x = copy.deepcopy(img)
        # 1st block
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
def go():
    device = torch.device('cuda')
    net = LeNet().to(device)
    y = net(torch.randn((224, 1, 28, 28)).to(device))
    return net

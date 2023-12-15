import torch
import torch.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=3, kernel_size=1, out_channels=64, dilation=6)
        self.layer2 = torch.nn.BatchNorm1d(num_features=64, eps=0.9221693585092791, momentum=0.7459598650898907, affine=True, track_running_stats=True)
        self.layer3 = torch.nn.ReLU(inplace=False)
        self.layer4 = torch.nn.Unfold(kernel_size=1, stride=(3, 1))
        self.layer5 = torch.nn.LogSigmoid()
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv1d(in_channels=128, kernel_size=1, out_channels=1024, padding_mode="circular")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


def go():
    model = pointnet()
    x = torch.randn(3, 3, 5)
    y = model(x)
    return model

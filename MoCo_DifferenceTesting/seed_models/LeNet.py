import copy
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv_2 = nn.Sigmoid()
        self.conv_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_4 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv_5 = nn.Sigmoid()
        self.conv_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(in_features=256, out_features=120)
        self.fc_2 = nn.Sigmoid()
        self.fc_3 = nn.Linear(in_features=120, out_features=84)
        self.fc_4 = nn.Sigmoid()
        self.fc_5 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        img = copy.deepcopy(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = x.view(img.shape[0], -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)
        x = self.fc_5(x)
        return x


if __name__ == '__main__':
    net = LeNet()
    x_tbd = torch.randn(3, 1, 28, 28)
    x_1 = torch.randn(3,)
    x_2 = torch.randn(3, 1)
    x_3 = torch.randn(3, 1, 28)
    x_4 = torch.randn(3, 1, 28, 28)
    x_5 = torch.randn(3, 1, 28, 28, 28)
    y = net(x_tbd).shape
    shape_list = []
    for i in range(len(y)):
        shape_list.append(y[i])
    print(str(shape_list))

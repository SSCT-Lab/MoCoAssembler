import torch
import torch.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        self.pool = nn.GroupNorm(num_groups=1, num_channels=1)

    def forward(self, x):
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.GroupNorm(num_groups=3, num_channels=1)(x)
        return x


if __name__ == '__main__':
    net = testnet()
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

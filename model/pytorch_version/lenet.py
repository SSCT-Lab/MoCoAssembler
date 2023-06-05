import torch
from torch import nn


class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        # 2nd block
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)

        # 3rd block
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)

        # 4th block
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    net = lenet()
    from torchsummary import summary
    summary(net, (3, 32, 32))

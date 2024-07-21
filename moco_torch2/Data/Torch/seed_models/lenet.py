import torch
import torch.nn as nn


class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=256, out_features=120)
        self.relu3 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=120, out_features=84)
        self.relu4 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)

        return x


def go():
    model = lenet()
    x = torch.randn([1, 1, 28, 28])
    y = model(x)
    return model

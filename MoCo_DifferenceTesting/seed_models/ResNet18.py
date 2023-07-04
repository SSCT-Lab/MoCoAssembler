import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.BatchNorm2d(num_features=64)
        self.conv1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    net = ResNet18()
    x_tbd = torch.randn(3, 3, 224, 224)
    x_1 = torch.randn(3,)
    x_2 = torch.randn(3, 3)
    x_3 = torch.randn(3, 3, 224)
    x_4 = torch.randn(3, 3, 224, 224)
    x_5 = torch.randn(3, 3, 224, 224, 224)
    y = net(x_tbd).shape
    shape_list = []
    for i in range(len(y)):
        shape_list.append(y[i])
    print(str(shape_list))


import torch
import torch.nn as nn


# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet50(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv1_3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=2048, out_features=1000)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1st block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        # 2nd block
        x = self.conv2(x)

        # 3rd block
        x = self.conv3(x)

        # 4th block
        x = self.conv4(x)

        # 5th block
        x = self.conv5(x)

        # output
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


def go():
    device = torch.device('cuda')
    res50 = ResNet50().to(device)
    xin = torch.randn((3, 3, 224, 224)).to(device)
    y = res50(xin)
    return res50

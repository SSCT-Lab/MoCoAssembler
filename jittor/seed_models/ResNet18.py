import jittor
import jittor.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # conv1_x
        self.conv1_1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = jittor.nn.BatchNorm2d(num_features=64)
        self.conv1_3 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])

        self.avgpool = jittor.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = jittor.nn.Linear(in_features=512, out_features=1000)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def execute(self, x):
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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            jittor.nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = jittor.nn.ReLU()

    def execute(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


def go():
    res18 = ResNet18()
    x = jittor.randn((2, 3, 224, 224))
    y = res18(x)
    return res18

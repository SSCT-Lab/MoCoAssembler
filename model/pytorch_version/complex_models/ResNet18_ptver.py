from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # conv1_x
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
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


if __name__ == '__main__':
    res18 = ResNet18(BasicBlock)
    print(res18)

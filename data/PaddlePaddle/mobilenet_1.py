import paddle
import paddle.nn as nn


class ConvBN(nn.Layer):  # 普通卷积层
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Depthwise(nn.Layer):  # 深度可分离卷积层
    def __init__(self, in_channels, out_channels, stride):
        super(Depthwise, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNet(nn.Layer):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile = nn.Sequential(
            ConvBN(3, 32, 2),
            Depthwise(32, 64, 1),
            Depthwise(64, 128, 2),
            Depthwise(128, 128, 1),
            Depthwise(128, 256, 2),
            Depthwise(256, 256, 1),
            Depthwise(256, 512, 2),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 1024, 2),
            Depthwise(1024, 1024, 1),
            nn.AvgPool2D(7, 7)
        )

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.mobile(x)
        out = paddle.reshape(out, [1, 1024])
        out = self.fc(out)

        return out


def mobilenet_1():
    return MobileNet()


if __name__ == '__main__':
    x = paddle.randn((1, 3, 224, 224))
    model = MobileNet()
    y = model(x)

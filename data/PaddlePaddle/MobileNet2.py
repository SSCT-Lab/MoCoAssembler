from numpy.core.shape_base import block
import paddle
import paddle.nn as nn

class block_m2(nn.Layer):
    def __init__(self, in_channels, out_channels, t=1, s=1):
        super(block_m2, self).__init__()

        self.convend = False

        self.conv1 = nn.Conv2D(in_channels, in_channels*t, kernel_size=1, stride=1)
        self.bn1   = nn.BatchNorm2D(in_channels*t)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2D(in_channels*t, in_channels, kernel_size=3, stride=s, padding=1, groups=in_channels)
        self.bn2   = nn.BatchNorm2D(in_channels)
        self.relu2 = nn.ReLU6()

        self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1)
        self.bn3   = nn.BatchNorm2D(out_channels)

        if in_channels == out_channels and s == 1:
            self.convend = nn.Conv2D(in_channels, in_channels, kernel_size=1)


    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if self.convend:
            out = self.convend(out+x)
        return self.bn3(out)


class MobileNetv2(nn.Layer):
    def __init__(self, num_classes):
        super(MobileNetv2, self).__init__()
            #   t   c  n  s
        cfg = [[1, 16, 1, 1],
               [6, 24, 1, 1],
               [6, 32, 3, 2],
               [6, 64, 4, 2],
               [6, 96, 3, 1],
               [6, 160, 3, 2],
               [6, 320, 1, 1]]

        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu1 = nn.ReLU()

        self.layers = self.make_layer(cfg, 32)

        self.conv2 = nn.Conv2D(320, 1280, kernel_size=1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(1280)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.AvgPool2D(7)
        self.conv3 = nn.Conv2D(1280, num_classes, kernel_size=1, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_classes)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(self.relu3(self.bn3(self.conv3(self.pool1(x)))))
        return self.softmax(x)

    def make_layer(self, layer_parameter, in_channels):
        layer = []
        for cfg in layer_parameter:
            for i in range(cfg[2]):
                stride = 1 if i != 0 else cfg[3]
                layer.append(block_m2(in_channels=in_channels, out_channels=cfg[1], t=cfg[0], s=stride))
                in_channels = cfg[1]
        return nn.Sequential(*layer)
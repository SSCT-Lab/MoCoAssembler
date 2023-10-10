import paddle
import paddle.nn as nn

class block_m1(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(block_m1, self).__init__()

        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.bn1   = nn.BatchNorm2D(in_channels)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=1)
        self.bn2   = nn.BatchNorm2D(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class MobileNetv1(nn.Layer):
    def __init__(self, num_classes):
        super(MobileNetv1, self).__init__()

        cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu1 = nn.ReLU()

        self.layers = self.make_layer(cfg, 32)
        self.pool1 = nn.AvgPool2D(7)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(1024, num_classes)
        self.relu2 = nn.ReLU()

        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.flatten(self.pool1(self.layers(x)))
        x = self.softmax(self.relu2(self.fc(x)))
        return x

    def make_layer(self, layer_parameter, in_channels):
        layer = []
        for cfg in layer_parameter:
            if isinstance(cfg, int):
                layer.append(block_m1(in_channels, out_channels=cfg))
                in_channels = cfg
            else:
                layer.append(block_m1(in_channels, out_channels=cfg[0], stride=cfg[1]))
                in_channels = cfg[0]
        return nn.Sequential(*layer)
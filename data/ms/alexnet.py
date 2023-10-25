import mindspore
import mindspore.nn as nn


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10, channel=3):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = nn.Conv2d(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = nn.Conv2d(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = nn.Conv2d(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = nn.Conv2d(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = mindspore.nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(6 * 6 * 256, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, num_classes)
        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def alexnet():
    return AlexNet()

import jittor as jt
import jittor.nn as nn
from jittorsummary import summary


class VGG_19(nn.Module):
    def __init__(self, class_num=1000):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4d = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5d = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, class_num)

        self.softmax = nn.Softmax(dim=1)

    def execute(self, x):
        # 1st block
        x = self.conv1a(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.relu(x)
        x = self.pool2(x)

        # 3rd block
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.conv3c(x)
        x = self.relu(x)
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.pool3(x)

        # 4th block
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.relu(x)
        x = self.conv4c(x)
        x = self.relu(x)
        x = self.conv4d(x)
        x = self.relu(x)
        x = self.pool4(x)

        # 5th block
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.relu(x)
        x = self.conv5c(x)
        x = self.relu(x)
        x = self.conv5d(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = jt.reshape(x, (-1, 512 * 7 * 7))

        # full connection
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = VGG_19()
    # input = jt.randn((1, 3, 224, 224))
    # output = net(input)
    summary(net, (3, 244, 244))
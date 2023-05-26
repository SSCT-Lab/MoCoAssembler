import jittor
import jittor.nn as nn
from jittorsummary import summary


class VGG_16(nn.Module):
    def __init__(self, class_num=1000):
        super().__init__()

        self.relu1a = jittor.nn.ReLU()
        self.relu1b = jittor.nn.ReLU()
        self.relu2a = jittor.nn.ReLU()
        self.relu2b = jittor.nn.ReLU()
        self.relu3a = jittor.nn.ReLU()
        self.relu3b = jittor.nn.ReLU()
        self.relu3c = jittor.nn.ReLU()
        self.relu4a = jittor.nn.ReLU()
        self.relu4b = jittor.nn.ReLU()
        self.relu4c = jittor.nn.ReLU()
        self.relu5a = jittor.nn.ReLU()
        self.relu5b = jittor.nn.ReLU()
        self.relu5c = jittor.nn.ReLU()
        self.relu6 = jittor.nn.ReLU()
        self.relu7 = jittor.nn.ReLU()

        self.conv1a = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = jittor.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = jittor.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2b = jittor.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = jittor.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3b = jittor.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3c = jittor.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = jittor.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4b = jittor.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4c = jittor.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5a = jittor.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5b = jittor.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5c = jittor.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = jittor.nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = jittor.nn.Linear(4096, 4096)
        self.fc8 = jittor.nn.Linear(4096, class_num)

        self.softmax = jittor.nn.Softmax(dim=1)

    def execute(self, x):
        # 1st block
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        x = self.pool2(x)

        # 3rd block
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)
        x = self.conv3c(x)
        x = self.relu3c(x)
        x = self.pool3(x)

        # 4th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)
        x = self.conv4c(x)
        x = self.relu4c(x)
        x = self.pool4(x)

        # 5th block
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)
        x = self.conv5c(x)
        x = self.relu5c(x)
        x = self.pool5(x)

        x = jittor.reshape(x, (-1, 512 * 7 * 7))

        # full connection
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = VGG_16()
    # input = jt.randn((4, 3, 224, 224))
    # output = net(input)
    summary(net, (3, 244, 244))
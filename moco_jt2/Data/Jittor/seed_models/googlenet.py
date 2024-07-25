import jittor.nn as nn
import jittor


class googlenet(nn.Module):
    def __init__(self):
        super(googlenet, self).__init__()

        self.relu1 = jittor.nn.ReLU()
        self.relu2 = jittor.nn.ReLU()
        self.relu3 = jittor.nn.ReLU()

        self.conv1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = jittor.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(in_channels=480, ch1x1=192,ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = jittor.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)

        self.avgpool1 = jittor.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = jittor.nn.Dropout(p=0.4)
        self.fc = jittor.nn.Linear(in_features=1024, out_features=1000)
        self.flatten = jittor.nn.Flatten()

    def execute(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool1(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.relu1 = jittor.nn.ReLU()
        self.relu2a = jittor.nn.ReLU()
        self.relu2b = jittor.nn.ReLU()
        self.relu3a = jittor.nn.ReLU()
        self.relu3b = jittor.nn.ReLU()
        self.relu4 = jittor.nn.ReLU()
        self.pool = jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.conv2a = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1)
        self.conv2b = jittor.nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)

        self.conv3a = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1)
        self.conv3b = jittor.nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)

        self.conv4 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1)
        self.cat = jittor.concat

    def execute(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)

        branch2 = self.conv2a(x)
        branch2 = self.relu2a(branch2)
        branch2 = self.conv2b(branch2)
        branch2 = self.relu2b(branch2)

        branch3 = self.conv3a(x)
        branch3 = self.relu3a(branch3)
        branch3 = self.conv3b(branch3)
        branch3 = self.relu3b(branch3)

        branch4 = self.conv4(x)
        branch4 = self.pool(branch4)
        branch4 = self.relu4(branch4)

        x = self.cat([branch1, branch2, branch3, branch4], dim=1)

        return x


def go():
    net = googlenet()
    x = jittor.randn((3, 3, 224, 224))
    y = net(x)
    return net

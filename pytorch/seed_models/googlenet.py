import torch.nn as nn
import torch


class googlenet(nn.Module):
    def __init__(self, class_num=1000):
        super(googlenet, self).__init__()

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool1 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
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

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.relu1 = torch.nn.ReLU()
        self.relu2a = torch.nn.ReLU()
        self.relu2b = torch.nn.ReLU()
        self.relu3a = torch.nn.ReLU()
        self.relu3b = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.conv2a = nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1)
        self.conv2b = nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1)
        self.conv3b = nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)

        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1)

    def forward(self, x):
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

        branch4 = self.pool(x)
        branch4 = self.conv4(branch4)
        branch4 = self.relu4(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        x = torch.cat(outputs, 1)

        return x


def go():
    net = googlenet().to('cuda')
    x = torch.randn((3, 3, 224, 224))
    x = x.to('cuda')
    y = net(x)
    return net

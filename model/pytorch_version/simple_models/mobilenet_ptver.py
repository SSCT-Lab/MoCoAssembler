import torch.nn as nn
from torchsummary import summary


class MobileNet(nn.Module):
    def __init__(self, in_channels):
        super(MobileNet, self).__init__()

        self.fc = nn.Linear(1024, 1000)
        self.relu = nn.ReLU(True)

        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)

        self.conv1a = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1b = nn.Conv2d(32, 64, 1, 1, 0)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.conv2b = nn.Conv2d(64, 128, 1, 1, 0)

        self.conv3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.conv3b = nn.Conv2d(128, 128, 1, 1, 0)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv4b = nn.Conv2d(128, 256, 1, 1, 0)

        self.conv5a = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv5b = nn.Conv2d(256, 256, 1, 1, 0)

        self.conv6a = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256)
        self.conv6b = nn.Conv2d(256, 512, 1, 1, 0)

        self.conv7a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv7b = nn.Conv2d(512, 512, 1, 1, 0)

        self.conv8a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv8b = nn.Conv2d(512, 512, 1, 1, 0)

        self.conv9a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv9b = nn.Conv2d(512, 512, 1, 1, 0)

        self.conv10a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv10b = nn.Conv2d(512, 512, 1, 1, 0)

        self.conv11a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv11b = nn.Conv2d(512, 512, 1, 1, 0)

        self.conv12a = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512)
        self.conv12b = nn.Conv2d(512, 1024, 1, 1, 0)

        self.conv13a = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=4, groups=1024)
        self.conv13b = nn.Conv2d(1024, 1024, 1, 1, 0)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 1st block
        x = self.conv0(x)

        # 2nd block
        x = self.conv1a(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.relu(x)

        # 3rd block
        x = self.conv2a(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.relu(x)

        # 4th block
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)

        # 5th block
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.relu(x)

        # 6th block
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.relu(x)

        # 7th block
        x = self.conv6a(x)
        x = self.relu(x)
        x = self.conv6b(x)
        x = self.relu(x)

        # 8th block
        x = self.conv7a(x)
        x = self.relu(x)
        x = self.conv7b(x)
        x = self.relu(x)

        # 9th block
        x = self.conv8a(x)
        x = self.relu(x)
        x = self.conv8b(x)
        x = self.relu(x)

        # 10th block
        x = self.conv9a(x)
        x = self.relu(x)
        x = self.conv9b(x)
        x = self.relu(x)

        # 11th block
        x = self.conv10a(x)
        x = self.relu(x)
        x = self.conv10b(x)
        x = self.relu(x)

        # 12th block
        x = self.conv11a(x)
        x = self.relu(x)
        x = self.conv11b(x)
        x = self.relu(x)

        # 13th block
        x = self.conv12a(x)
        x = self.relu(x)
        x = self.conv12b(x)
        x = self.relu(x)

        # 14th block
        x = self.conv13a(x)
        x = self.relu(x)
        x = self.conv13b(x)
        x = self.relu(x)

        # 15th block
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        # output
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNet(3)
    summary(model, (3, 224, 224))

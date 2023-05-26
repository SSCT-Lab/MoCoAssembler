import jittor
import jittor as jt
import jittor.nn as nn
from jittorsummary import summary

class MobileNet(nn.Module):
    def __init__(self, in_channels):
        super(MobileNet, self).__init__()

        self.fc = jittor.nn.Linear(1024, 1000)
        self.relu1a = jittor.nn.ReLU()
        self.relu1b = jittor.nn.ReLU()
        self.relu2a = jittor.nn.ReLU()
        self.relu2b = jittor.nn.ReLU()
        self.relu3a = jittor.nn.ReLU()
        self.relu3b = jittor.nn.ReLU()
        self.relu4a = jittor.nn.ReLU()
        self.relu4b = jittor.nn.ReLU()
        self.relu5a = jittor.nn.ReLU()
        self.relu5b = jittor.nn.ReLU()
        self.relu6a = jittor.nn.ReLU()
        self.relu6b = jittor.nn.ReLU()
        self.relu7a = jittor.nn.ReLU()
        self.relu7b = jittor.nn.ReLU()
        self.relu8a = jittor.nn.ReLU()
        self.relu8b = jittor.nn.ReLU()
        self.relu9a = jittor.nn.ReLU()
        self.relu9b = jittor.nn.ReLU()
        self.relu10a = jittor.nn.ReLU()
        self.relu10b = jittor.nn.ReLU()
        self.relu11a = jittor.nn.ReLU()
        self.relu11b = jittor.nn.ReLU()
        self.relu12a = jittor.nn.ReLU()
        self.relu12b = jittor.nn.ReLU()
        self.relu13a = jittor.nn.ReLU()
        self.relu13b = jittor.nn.ReLU()

        self.conv0 = jittor.nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)

        self.conv1a = jittor.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1b = jittor.nn.Conv2d(32, 64, 1, 1, 0)

        self.conv2a = jittor.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.conv2b = jittor.nn.Conv2d(64, 128, 1, 1, 0)

        self.conv3a = jittor.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.conv3b = jittor.nn.Conv2d(128, 128, 1, 1, 0)

        self.conv4a = jittor.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv4b = jittor.nn.Conv2d(128, 256, 1, 1, 0)

        self.conv5a = jittor.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv5b = jittor.nn.Conv2d(256, 256, 1, 1, 0)

        self.conv6a = jittor.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256)
        self.conv6b = jittor.nn.Conv2d(256, 512, 1, 1, 0)

        self.conv7a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv7b = jittor.nn.Conv2d(512, 512, 1, 1, 0)

        self.conv8a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv8b = jittor.nn.Conv2d(512, 512, 1, 1, 0)

        self.conv9a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv9b = jittor.nn.Conv2d(512, 512, 1, 1, 0)

        self.conv10a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv10b = jittor.nn.Conv2d(512, 512, 1, 1, 0)

        self.conv11a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv11b = jittor.nn.Conv2d(512, 512, 1, 1, 0)

        self.conv12a = jittor.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512)
        self.conv12b = jittor.nn.Conv2d(512, 1024, 1, 1, 0)

        self.conv13a = jittor.nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=4, groups=1024)
        self.conv13b = jittor.nn.Conv2d(1024, 1024, 1, 1, 0)

        self.pool = jittor.nn.AdaptiveAvgPool2d(1)

    def execute(self, x):
        # 1st block
        x = self.conv0(x)

        # 2nd block
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)

        # 3rd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)

        # 4th block
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)

        # 5th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)

        # 6th block
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)

        # 7th block
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.conv6b(x)
        x = self.relu6b(x)

        # 8th block
        x = self.conv7a(x)
        x = self.relu7a(x)
        x = self.conv7b(x)
        x = self.relu7b(x)

        # 9th block
        x = self.conv8a(x)
        x = self.relu8a(x)
        x = self.conv8b(x)
        x = self.relu8b(x)

        # 10th block
        x = self.conv9a(x)
        x = self.relu9a(x)
        x = self.conv9b(x)
        x = self.relu9b(x)

        # 11th block
        x = self.conv10a(x)
        x = self.relu10a(x)
        x = self.conv10b(x)
        x = self.relu10b(x)

        # 12th block
        x = self.conv11a(x)
        x = self.relu11a(x)
        x = self.conv11b(x)
        x = self.relu11b(x)

        # 13th block
        x = self.conv12a(x)
        x = self.relu12a(x)
        x = self.conv12b(x)
        x = self.relu12b(x)

        # 14th block
        x = self.conv13a(x)
        x = self.relu13a(x)
        x = self.conv13b(x)
        x = self.relu13b(x)

        # 15th block
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        # output
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = MobileNet(3)
    summary(model, (3, 224, 224))
    # x = jittor.randn((4, 3, 224, 224))
    # y = model(x)

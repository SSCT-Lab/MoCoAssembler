import torch
import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.relu1a = nn.ReLU()
        self.relu1b = nn.ReLU()
        self.relu2a = nn.ReLU()
        self.relu2b = nn.ReLU()
        self.relu3a = nn.ReLU()
        self.relu3b = nn.ReLU()
        self.relu4a = nn.ReLU()
        self.relu4b = nn.ReLU()
        self.relu5a = nn.ReLU()
        self.relu5b = nn.ReLU()
        self.relu6a = nn.ReLU()
        self.relu6b = nn.ReLU()
        self.relu7a = nn.ReLU()
        self.relu7b = nn.ReLU()
        self.relu8a = nn.ReLU()
        self.relu8b = nn.ReLU()
        self.relu9a = nn.ReLU()
        self.relu9b = nn.ReLU()
        self.relu10a = nn.ReLU()
        self.relu10b = nn.ReLU()
        self.relu11a = nn.ReLU()
        self.relu11b = nn.ReLU()
        self.relu12a = nn.ReLU()
        self.relu12b = nn.ReLU()
        self.relu13a = nn.ReLU()
        self.relu13b = nn.ReLU()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.conv1a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv4a = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv4b = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv5b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv6a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256)
        self.conv6b = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv7a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv7b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv8a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv8b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv9a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv9b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv10a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv10b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv11a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv11b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv12a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, groups=512)
        self.conv12b = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.conv13a = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=4, groups=1024)
        self.conv13b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.conv6b(x)
        x = self.relu6b(x)
        x = self.conv7a(x)
        x = self.relu7a(x)
        x = self.conv7b(x)
        x = self.relu7b(x)
        x = self.conv8a(x)
        x = self.relu8a(x)
        x = self.conv8b(x)
        x = self.relu8b(x)
        x = self.conv9a(x)
        x = self.relu9a(x)
        x = self.conv9b(x)
        x = self.relu9b(x)
        x = self.conv10a(x)
        x = self.relu10a(x)
        x = self.conv10b(x)
        x = self.relu10b(x)
        x = self.conv11a(x)
        x = self.relu11a(x)
        x = self.conv11b(x)
        x = self.relu11b(x)
        x = self.conv12a(x)
        x = self.relu12a(x)
        x = self.conv12b(x)
        x = self.relu12b(x)
        x = self.conv13a(x)
        x = self.relu13a(x)
        x = self.conv13b(x)
        x = self.relu13b(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = MobileNet()
    x_tbd = torch.randn(3, 3, 224, 224)
    x_1 = torch.randn(3,)
    x_2 = torch.randn(3, 3)
    x_3 = torch.randn(3, 3, 224)
    x_4 = torch.randn(3, 3, 224, 224)
    x_5 = torch.randn(3, 3, 224, 224, 224)
    y = net(x_tbd).shape
    shape_list = []
    for i in range(len(y)):
        shape_list.append(y[i])
    print(str(shape_list))

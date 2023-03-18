import torch
import torch.nn as nn
from torchsummary import summary


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)

        self.conv2a = nn.Conv2d(96, 16, kernel_size=1, stride=1)
        self.conv2b = nn.Conv2d(16, 64, kernel_size=1, stride=1)
        self.conv2c = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 16, kernel_size=1, stride=1)

        self.conv4a = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.conv4b = nn.Conv2d(32, 128, kernel_size=1, stride=1)
        self.conv4c = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(128, 32, kernel_size=1, stride=1)

        self.conv6a = nn.Conv2d(128, 48, kernel_size=1, stride=1)
        self.conv6b = nn.Conv2d(48, 192, kernel_size=1, stride=1)
        self.conv6c = nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(192, 48, kernel_size=1, stride=1)

        self.conv8a = nn.Conv2d(192, 64, kernel_size=1, stride=1)
        self.conv8b = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.conv8c = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(256, 64, kernel_size=1, stride=1)

        self.conv10 = nn.Conv2d(256, 5, kernel_size=1, stride=1)


    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.pool(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu(x)
        y1 = self.conv2b(x)
        y1 = self.relu(y1)
        y2 = self.conv2c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)

        # 3rd block
        x = self.conv3(x)
        x = self.relu(x)
        y1 = self.conv2b(x)
        y1 = self.relu(y1)
        y2 = self.conv2c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)

        # 4th block
        x = self.conv4a(x)
        x = self.relu(x)
        y1 = self.conv4b(x)
        y1 = self.relu(y1)
        y2 = self.conv4c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)
        x = self.pool(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu(x)
        y1 = self.conv4b(x)
        y1 = self.relu(y1)
        y2 = self.conv4c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)

        # 6th block
        x = self.conv6a(x)
        x = self.relu(x)
        y1 = self.conv6b(x)
        y1 = self.relu(y1)
        y2 = self.conv6c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)

        # 7th block
        x = self.conv7(x)
        x = self.relu(x)
        y1 = self.conv6b(x)
        y1 = self.relu(y1)
        y2 = self.conv6c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)

        # 8th block
        x = self.conv8a(x)
        x = self.relu(x)
        y1 = self.conv8b(x)
        y1 = self.relu(y1)
        y2 = self.conv8c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)
        x = self.pool(x)

        # 9th block
        x = self.conv9(x)
        x = self.relu(x)
        y1 = self.conv8b(x)
        y1 = self.relu(y1)
        y2 = self.conv8c(x)
        y2 = self.relu(y2)
        x = torch.concat([y1, y2], dim=-1)
        x = self.pool(x)

        # 10th block
        x = self.conv10(x)
        x = self.avgpool(x)

        # output
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = SqueezeNet()
    summary(model, (3, 244, 244))

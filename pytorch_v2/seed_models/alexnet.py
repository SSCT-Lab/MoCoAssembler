import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.relu5 = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU()
        self.relu7 = torch.nn.ReLU()

        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=6)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.linear1 = torch.nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = torch.nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 3rd block
        x = self.conv3(x)
        x = self.relu3(x)

        # 4th block
        x = self.conv4(x)
        x = self.relu4(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        # 6th block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 7th block
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)

        # 8th block
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)

        # output
        x = self.linear3(x)

        return x


def go():
    device = torch.device('cuda')
    model = AlexNet().to(device)
    x = torch.randn((6, 3, 224, 224)).to(device)
    y = model(x)

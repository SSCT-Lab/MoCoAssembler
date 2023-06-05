import torch
import torch.nn as nn
from torchsummary import summary


class alexnet(nn.Module):
    def __init__(self, class_num: int = 1000, dropout: float = 0.5) -> None:
        super(alexnet, self).__init__()

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=class_num)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 2nd block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 3rd block
        x = self.conv3(x)
        x = self.relu(x)

        # 4th block
        x = self.conv4(x)
        x = self.relu(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        # 6th block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 7th block
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        # 8th block
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        # output
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = alexnet()
    summary(model, (3, 224, 224))

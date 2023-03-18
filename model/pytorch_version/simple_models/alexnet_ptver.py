import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.relu()

        # 4th block
        x = self.conv4(x)
        x = self.relu()

        # 5th block
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        # 6th block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 7th block
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu()

        # 8th block
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu()

        # output
        x = self.linear3(x)

        return x


if __name__ == '__main__':
    model = AlexNet()
    summary(model, (3, 224, 224))

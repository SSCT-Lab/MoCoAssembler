import torch
import torch.nn as nn
from torchsummary import summary


class DenseNet(nn.Module):
    def __init__(self, density) -> None:
        super().__init__()

        self.density = density

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv64to128 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.conv32to128 = nn.Conv2d(32, 128, kernel_size=1, stride=1)
        self.conv128to32 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv64to32 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=1)
        self.conv32to256 = nn.Conv2d(32, 256, kernel_size=1, stride=1)
        self.conv32to512 = nn.Conv2d(32, 512, kernel_size=1, stride=1)
        self.conv256to128 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1)
        self.conv512to128 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features_list = []

        # 1st block
        x = self.conv1(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 2nd block
        x = self.bn64(x)
        x = self.relu(x)
        x = self.conv64to128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv128to32(x)
        x = self.dropout(x)
        features_list.append(x)
        x = torch.concat(features_list, dim=-1)
        for i in range(1, self.density[0]):
            x = self.bn32(x)
            x = self.relu(x)
            x = self.conv32to128(x)
            x = self.bn128(x)
            x = self.relu(x)
            x = self.conv128to32(x)
            x = self.dropout(x)
            features_list.append(x)
            x = torch.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv32to128(x)
        x = self.pool2(x)

        # 3rd block
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv128to32(x)
        x = self.dropout(x)
        features_list.append(x)
        x = torch.concat(features_list, dim=-1)
        for i in range(1, self.density[1]):
            x = self.bn32(x)
            x = self.relu(x)
            x = self.conv32to128(x)
            x = self.bn128(x)
            x = self.relu(x)
            x = self.conv128to32(x)
            x = self.dropout(x)
            features_list.append(x)
            x = torch.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv32to256(x)
        x = self.pool2(x)

        # 4th block
        x = self.bn256(x)
        x = self.relu(x)
        x = self.conv256to128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv128to32(x)
        x = self.dropout(x)
        features_list.append(x)
        x = torch.concat(features_list, dim=-1)
        for i in range(1, self.density[2]):
            x = self.bn32(x)
            x = self.relu(x)
            x = self.conv32to128(x)
            x = self.bn128(x)
            x = self.relu(x)
            x = self.conv128to32(x)
            x = self.dropout(x)
            features_list.append(x)
            x = torch.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv32to512(x)
        x = self.pool2(x)

        # 5th block
        x = self.bn512(x)
        x = self.relu(x)
        x = self.conv512to128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv128to32(x)
        x = self.dropout(x)
        features_list.append(x)
        x = torch.concat(features_list, dim=-1)
        for i in range(1, self.density[3]):
            x = self.bn32(x)
            x = self.relu(x)
            x = self.conv32to128(x)
            x = self.bn128(x)
            x = self.relu(x)
            x = self.conv128to32(x)
            x = self.dropout(x)
            features_list.append(x)
            x = torch.concat(features_list, dim=-1)
        features_list.clear()

        x = self.avgpool(x)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    # model121 = DenseNet([6, 12, 24, 16])
    model_simplified = DenseNet([1, 2, 4, 3])
    summary(model_simplified, (3, 224, 224))

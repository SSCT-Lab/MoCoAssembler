import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=6)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features=9216, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    net = AlexNet()
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

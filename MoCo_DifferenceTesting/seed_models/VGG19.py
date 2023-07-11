import torch
import torch as jt
import torch.nn as nn


class VGG_19(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu1a = nn.ReLU()
        self.relu1b = nn.ReLU()
        self.relu2a = nn.ReLU()
        self.relu2b = nn.ReLU()
        self.relu3a = nn.ReLU()
        self.relu3b = nn.ReLU()
        self.relu3c = nn.ReLU()
        self.relu3d = nn.ReLU()
        self.relu4a = nn.ReLU()
        self.relu4b = nn.ReLU()
        self.relu4c = nn.ReLU()
        self.relu4d = nn.ReLU()
        self.relu5a = nn.ReLU()
        self.relu5b = nn.ReLU()
        self.relu5c = nn.ReLU()
        self.relu5d = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4d = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5d = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Linear(in_features=25088, out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=1000)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)
        x = self.conv3c(x)
        x = self.relu3c(x)
        x = self.conv3d(x)
        x = self.relu3d(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)
        x = self.conv4c(x)
        x = self.relu4c(x)
        x = self.conv4d(x)
        x = self.relu4d(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)
        x = self.conv5c(x)
        x = self.relu5c(x)
        x = self.conv5d(x)
        x = self.relu5d(x)
        x = self.pool5(x)
        x = jt.reshape(x, (-1, 512 * 7 * 7))
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = VGG_19()
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

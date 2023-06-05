import torch
from torch import nn


class simp(nn.Module):
    def __init__(self):
        super(simp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(16*14*14, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)


        return x


if __name__ == '__main__':
    net = simp()
    from torchsummary import summary
    summary(net, (3, 32, 32))

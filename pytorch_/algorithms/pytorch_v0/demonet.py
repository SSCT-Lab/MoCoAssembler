import torch.nn as nn
from torchsummary import summary


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.CELU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(3)
        self.linear = nn.Bilinear(4, 4, 2)
        self.dropout = nn.Dropout(p=0.5)

        self.conv = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(2, eps=1.1, momentum=4.1, affine=False, track_running_stats=False)

    def forward(self, x):
        print("1", x.shape)
        # x = self.conv(x)
        # print("2", x.shape)
        # x = self.batchnorm(x)
        x = self.avgpool(x)
        print("3", x.shape)
        return x


if __name__ == '__main__':
    model = DemoNet()
    summary(model, (2, 3, 4))

    # x = torch.tensor([1, 2, 3, 4])  # x.dim()=1
    # print(x)
    # print(x.shape)
    # y = x.unsqueeze(0)
    # print(y)
    # print(y.shape)  # 此时y.dim()=2
    # z = x.unsqueeze(1)
    # print(z)
    # print(z.shape)  # 此时z.dim()=2

    # x = torch.empty(2, 3)
    # nn.init.eye_(x)
    # print(x)

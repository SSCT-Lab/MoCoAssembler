import torch
import torch.nn as nn
from torchsummary import summary


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(6)

        self.linear = nn.Linear(3, 2)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("shape0=" + str(x.shape))
        #
        # x = self.linear(x)
        #
        # print("shape5=" + str(x.shape))

        x = torch.ones(2, 3)
        print(x)
        x = self.dropout(x)
        print(x)
        # print("shape6=" + str(x.shape))

        return x


if __name__ == '__main__':
    model = DemoNet()
    summary(model, (3, 2, 3))

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

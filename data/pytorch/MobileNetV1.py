import torch


def BottleneckV1(in_channels, out_channels, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU6(inplace=True),
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU6(inplace=True)
    )


class MobileNetV1(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(inplace=True),
        )

        self.bottleneck = torch.nn.Sequential(
            BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=2),
            BottleneckV1(128, 128, stride=1),
            BottleneckV1(128, 256, stride=2),
            BottleneckV1(256, 256, stride=1),
            BottleneckV1(256, 512, stride=2),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 1024, stride=2),
            BottleneckV1(1024, 1024, stride=1),
        )

        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = torch.nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.softmax = torch.nn.Softmax(dim=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    model = MobileNetV1()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

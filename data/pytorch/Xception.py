import torch


def ConvBN(in_channels, out_channels, kernel_size, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=0 if kernel_size == 1 else (kernel_size - 1) // 2),
        torch.nn.BatchNorm2d(out_channels),
    )


def ConvBNRelu(in_channels, out_channels, kernel_size, stride):
    return torch.nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size, stride),
        torch.nn.ReLU6(inplace=True),
    )


def SeparableConvolution(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                        groups=in_channels),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    )


def SeparableConvolutionRelu(in_channels, out_channels):
    return torch.nn.Sequential(
        SeparableConvolution(in_channels, out_channels),
        torch.nn.ReLU6(inplace=True),
    )


def ReluSeparableConvolution(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.ReLU6(inplace=True),
        SeparableConvolution(in_channels, out_channels)
    )


class EntryBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        mid_channels = out_channels

        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.bottleneck = torch.nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels,
                                     out_channels=mid_channels) if first_relu else SeparableConvolution(
                in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class MiddleBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels

        self.bottleneck = torch.nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        return out + x


class ExitBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels

        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.bottleneck = torch.nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class Xception(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        self.entryFlow = torch.nn.Sequential(
            ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            EntryBottleneck(in_channels=64, out_channels=128, first_relu=False),
            EntryBottleneck(in_channels=128, out_channels=256, first_relu=True),
            EntryBottleneck(in_channels=256, out_channels=728, first_relu=True),
        )
        self.middleFlow = torch.nn.Sequential(
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
        )
        self.exitFlow = torch.nn.Sequential(
            ExitBottleneck(in_channels=728, out_channels=1024),
            SeparableConvolutionRelu(in_channels=1024, out_channels=1536),
            SeparableConvolutionRelu(in_channels=1536, out_channels=2048),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.linear = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entryFlow(x)
        x = self.middleFlow(x)
        x = self.exitFlow(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


if __name__ == '__main__':
    model = Xception()
    print(model)

    input = torch.randn(1, 3, 299, 299)
    output = model(input)
    print(output.shape)

import torch


class ResNeXtBlock(torch.nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=2, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1,
                            bias=False,
                            groups=cardinality),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1,
                                stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    model = ResNeXtBlock(in_places=256, places=128)
    print(model)

    input = torch.randn(1, 256, 64, 64)
    out = model(input)
    print(out.shape)

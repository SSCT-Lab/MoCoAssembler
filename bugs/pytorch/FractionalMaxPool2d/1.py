import torch
import torch.nn as nn


class googlenet(nn.Module):
    def __init__(self):
        super(googlenet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=(7, 4))
        self.layer2 = torch.nn.SELU(inplace=False)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=(1, 4), ceil_mode=True)
        self.layer4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding="same")
        self.layer7 = torch.nn.ReLU(inplace=False)
        self.layer8 = torch.nn.FractionalMaxPool2d(kernel_size=3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


class inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(inception, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)
        self.layer6 = torch.nn.ReLU()
        self.layer7 = torch.nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1)
        self.layer8 = torch.nn.ReLU()
        self.layer9 = torch.nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)
        self.layer10 = torch.nn.ReLU()
        self.layer11 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer12 = torch.nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1)
        self.layer13 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch2 = self.layer3(x)
        branch2 = self.layer4(branch2)
        branch2 = self.layer5(branch2)
        branch2 = self.layer6(branch2)
        branch3 = self.layer7(x)
        branch3 = self.layer8(branch3)
        branch3 = self.layer9(branch3)
        branch3 = self.layer10(branch3)
        branch4 = self.layer11(x)
        branch4 = self.layer12(branch4)
        branch4 = self.layer13(branch4)
        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


def go():
    model = googlenet()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model

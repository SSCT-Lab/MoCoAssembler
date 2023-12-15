import torch
import torch.nn as nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.layer1 = torch.nn.AvgPool2d(kernel_size=11, stride=4, padding=2)
        self.layer2 = torch.nn.SELU(inplace=True)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=4)
        self.layer4 = torch.nn.ZeroPad2d(padding=1)
        self.layer5 = torch.nn.LeakyReLU(inplace=False)
        self.layer6 = torch.nn.ZeroPad2d(padding=(5, 1, 4, 7))
        self.layer7 = torch.nn.ZeroPad2d(padding=1)
        self.layer8 = torch.nn.LeakyReLU(inplace=True)
        self.layer9 = torch.nn.ReflectionPad2d(padding=1)
        self.layer10 = torch.nn.LeakyReLU(inplace=False)
        self.layer11 = torch.nn.Unfold(kernel_size=(3, 3), stride=1, padding=0)
        self.layer12 = torch.nn.ConstantPad2d(padding=2, value=1)
        self.layer13 = torch.nn.AdaptiveAvgPool3d(output_size=5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        return x


def go():
    model = alexnet()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model

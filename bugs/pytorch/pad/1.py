import torch
import torch.nn as nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.layer1 = torch.nn.AvgPool2d(kernel_size=11, stride=4, padding=2)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, return_indices=False)
        self.layer4 = torch.nn.ReflectionPad2d(padding=2)
        self.layer5 = torch.nn.ReLU6(inplace=True)
        self.layer6 = torch.nn.ZeroPad2d(padding=(1, 6))
        self.layer7 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


def go():
    model = alexnet()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model

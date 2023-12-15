import torch
import torch.nn as nn


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.layer2 = torch.nn.ReplicationPad2d(padding=1)
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.AvgPool2d(kernel_size=1, stride=(2, 3), padding=0)
        self.layer5 = torch.nn.ELU(alpha=0.9619829754955781, inplace=True)
        self.layer6 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=6, return_indices=False, ceil_mode=True)
        self.layer7 = torch.nn.ReLU(inplace=False)
        self.layer8 = torch.nn.ReplicationPad2d(padding=0)
        self.layer9 = torch.nn.ReLU(inplace=False)
        self.layer10 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=False, divisor_override=8)
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.ReflectionPad2d(padding=0)
        self.layer13 = torch.nn.SELU(inplace=True)
        self.layer14 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=True, divisor_override=1)
        self.layer15 = torch.nn.SELU(inplace=True)
        self.layer16 = torch.nn.Unfold(kernel_size=1, stride=1, padding=0, dilation=(2, 2))
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256, padding_mode="circular")

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
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        return x


def go():
    model = mobilenet()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model

import torch
import torch.nn as nn


class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        self.layer1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.layer2 = torch.nn.ReLU(inplace=False)
        self.layer3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=4, padding=1)
        self.layer4 = torch.nn.ReLU(inplace=False)
        self.layer5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.layer6 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=(1, 5), return_indices=False, ceil_mode=True)
        self.layer7 = torch.nn.ReLU(inplace=True)
        self.layer8 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=6, return_indices=False, ceil_mode=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.ConstantPad2d(padding=(7, 8), value=193)
        self.layer11 = torch.nn.ZeroPad2d(padding=1)
        self.layer12 = torch.nn.ReLU6(inplace=True)
        self.layer13 = torch.nn.AdaptiveAvgPool3d(output_size=1)
        self.layer14 = torch.nn.SELU(inplace=False)
        self.layer15 = torch.nn.MultiheadAttention(embed_dim=-1, num_heads=1)

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
        return x


def go():
    model = vgg19()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model
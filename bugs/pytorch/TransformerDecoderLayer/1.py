import torch
import torch.nn as nn


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, dilation=(6, 4))
        self.layer2 = torch.nn.BatchNorm2d(num_features=64, affine=True)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer4 = InceptionA_59(in_channels=64, out_channels=64, stride=1)
        self.layer5 = InceptionA_70(in_channels=64, out_channels=64, stride=1)
        self.layer6 = InceptionB_198(in_channels=64, out_channels=128, stride=2)
        self.layer7 = InceptionA_499(in_channels=128, out_channels=128, stride=1)
        self.layer8 = InceptionB_1311(in_channels=128, out_channels=256, stride=2)
        self.layer9 = InceptionA_2893(in_channels=256, out_channels=256, stride=1)
        self.layer10 = InceptionB_4667(in_channels=256, out_channels=512, stride=2)
        self.layer11 = InceptionA_8490(in_channels=512, out_channels=512, stride=1)
        self.layer12 = torch.nn.TransformerDecoderLayer(d_model=1, nhead=0)

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
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


class InceptionB(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionB, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer6 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.layer7 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer9 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        branch2 = self.layer6(x)
        branch2 = self.layer7(branch2)
        x = branch1+ branch2
        x = self.layer9(x)
        return x


class InceptionA_59(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_59, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


class InceptionA_70(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_70, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


class InceptionB_198(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionB_198, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer6 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.layer7 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer9 = torch.nn.Hardsigmoid(inplace=False)

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        branch2 = self.layer6(x)
        branch2 = self.layer7(branch2)
        x = branch1+ branch2
        x = self.layer9(x)
        return x


class InceptionA_499(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_499, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


class InceptionB_1311(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionB_1311, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer6 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.layer7 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer9 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        branch2 = self.layer6(x)
        branch2 = self.layer7(branch2)
        x = branch1+ branch2
        x = self.layer9(x)
        return x


class InceptionA_2893(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_2893, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.LogSigmoid()
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


class InceptionB_4667(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionB_4667, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer6 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.layer7 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer9 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        branch2 = self.layer6(x)
        branch2 = self.layer7(branch2)
        x = branch1+ branch2
        x = self.layer9(x)
        return x


class InceptionA_8490(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_8490, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = torch.nn.ReLU()

    def forward(self, x):
        branch1 = self.layer1(x)
        branch1 = self.layer2(branch1)
        branch1 = self.layer3(branch1)
        branch1 = self.layer4(branch1)
        branch1 = self.layer5(branch1)
        x = branch1+ x
        x = self.layer7(x)
        return x


def go():
    model = resnet18()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model

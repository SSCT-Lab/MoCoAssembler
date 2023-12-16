import jittor
import jittor.nn as nn


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, dilation=(2, 4))
        self.layer2 = jittor.nn.BatchNorm2d(num_features=64, affine=False)
        self.layer3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer4 = InceptionA_120(in_channels=64, out_channels=64, stride=1)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = jittor.nn.ReLU()

    def execute(self, x):
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
        self.layer1 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer6 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.layer7 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer9 = jittor.nn.ReLU()

    def execute(self, x):
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


class InceptionA_120(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InceptionA_120, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5 = jittor.nn.BatchNorm2d(num_features=out_channels)
        self.layer7 = jittor.nn.ConstantPad2d(padding=-1, value=0)

    def execute(self, x):
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
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

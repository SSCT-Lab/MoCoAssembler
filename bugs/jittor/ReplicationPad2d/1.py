import jittor
import jittor.nn as nn


class googlenet(nn.Module):
    def __init__(self):
        super(googlenet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=(5, 8), padding=3)
        self.layer2 = jittor.nn.Leaky_relu()
        self.layer3 = jittor.nn.Tanh()
        self.layer4 = jittor.nn.Conv(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.layer5 = jittor.nn.ReLU6()
        self.layer6 = jittor.nn.Sigmoid()
        self.layer7 = jittor.nn.ReplicationPad2d(padding=-1)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


class inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(inception, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1)
        self.layer4 = jittor.nn.ReLU()
        self.layer5 = jittor.nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)
        self.layer6 = jittor.nn.ReLU()
        self.layer7 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1)
        self.layer8 = jittor.nn.ReLU()
        self.layer9 = jittor.nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)
        self.layer10 = jittor.nn.ReLU()
        self.layer11 = jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer12 = jittor.nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1)
        self.layer13 = jittor.nn.ReLU()

    def execute(self, x):
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
        x = jittor.cat([branch1, branch2, branch3, branch4], dim=1)
        return x


def go():
    model = googlenet()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

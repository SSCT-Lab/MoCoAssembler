import jittor
import jittor.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.layer1 = jittor.nn.Conv1d(in_channels=3, kernel_size=1, out_channels=64)
        self.layer2 = jittor.nn.BatchNorm1d(num_features=64)
        self.layer3 = jittor.nn.ReLU()
        self.layer4 = jittor.nn.Conv1d(in_channels=64, kernel_size=1, out_channels=128)
        self.layer5 = jittor.nn.BatchNorm1d(num_features=128)
        self.layer6 = jittor.nn.ReLU()
        self.layer7 = jittor.nn.Conv1d(in_channels=128, kernel_size=1, out_channels=1024)
        self.layer8 = jittor.nn.BatchNorm1d(num_features=1024)
        self.layer9 = jittor.nn.ReLU()
        self.layer10 = jittor.nn.Flatten()
        self.layer11 = jittor.nn.Linear(in_features=5120, out_features=512)
        self.layer12 = jittor.nn.ReLU()
        self.layer13 = jittor.nn.Linear(in_features=512, out_features=256)
        self.layer14 = jittor.nn.ReLU()
        self.layer15 = jittor.nn.Linear(in_features=256, out_features=10)

    def execute(self, x):
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
    model = pointnet()
    x = jittor.randn(3, 3, 5)
    y = model(x)
    return model

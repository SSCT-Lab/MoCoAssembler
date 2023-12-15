import torch
import torch.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=3, kernel_size=1, out_channels=64)
        self.layer2 = torch.nn.BatchNorm1d(num_features=64)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Conv1d(in_channels=64, kernel_size=1, out_channels=128)
        self.layer5 = torch.nn.BatchNorm1d(num_features=128)
        self.layer6 = torch.nn.ReLU()
        self.layer7 = torch.nn.Conv1d(in_channels=128, kernel_size=1, out_channels=1024)
        self.layer8 = torch.nn.BatchNorm1d(num_features=1024)
        self.layer9 = torch.nn.ReLU()
        self.layer10 = torch.nn.Flatten()
        self.layer11 = torch.nn.Linear(in_features=5120, out_features=512)
        self.layer12 = torch.nn.ReLU()
        self.layer13 = torch.nn.Linear(in_features=512, out_features=256)
        self.layer14 = torch.nn.ReLU()
        self.layer15 = torch.nn.Linear(in_features=256, out_features=10)

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
    model = pointnet()
    x = torch.randn(3, 3, 5)
    y = model(x)
    return model

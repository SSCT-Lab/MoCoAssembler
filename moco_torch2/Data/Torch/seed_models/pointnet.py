import torch
import torch.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn4 = torch.nn.BatchNorm1d(num_features=128, momentum=0.9)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bn5 = torch.nn.BatchNorm1d(num_features=1024, momentum=0.9)
        self.relu5 = torch.nn.ReLU()
        self.globalpool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.bn6 = torch.nn.BatchNorm1d(num_features=512, momentum=0.9)
        self.relu6 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=512, out_features=256)
        self.bn7 = torch.nn.BatchNorm1d(num_features=256, momentum=0.9)
        self.relu7 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.linear2(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.linear3(x)

        return x


def go():
    model = pointnet().to('cuda')
    x = torch.randn([2, 3, 2048]).to('cuda')
    y = model(x)
    return model

import mxnet
import mxnet.gluon.nn as nn


class pointnet(nn.Module):
    def __init__(self):
        super(pointnet, self).__init__()
        self.conv1 = mxnet.gluon.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.bn1 = mxnet.gluon.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu1 = mxnet.gluon.nn.ReLU()
        self.conv2 = mxnet.gluon.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn2 = mxnet.gluon.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu2 = mxnet.gluon.nn.ReLU()
        self.conv3 = mxnet.gluon.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn3 = mxnet.gluon.nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.relu3 = mxnet.gluon.nn.ReLU()
        self.conv4 = mxnet.gluon.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn4 = mxnet.gluon.nn.BatchNorm1d(num_features=128, momentum=0.9)
        self.relu4 = mxnet.gluon.nn.ReLU()
        self.conv5 = mxnet.gluon.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bn5 = mxnet.gluon.nn.BatchNorm1d(num_features=1024, momentum=0.9)
        self.relu5 = mxnet.gluon.nn.ReLU()
        self.globalpool = mxnet.gluon.nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = mxnet.gluon.nn.Flatten()
        self.linear1 = mxnet.gluon.nn.Linear(in_features=1024, out_features=512)
        self.bn6 = mxnet.gluon.nn.BatchNorm1d(num_features=512, momentum=0.9)
        self.relu6 = mxnet.gluon.nn.ReLU()
        self.linear2 = mxnet.gluon.nn.Linear(in_features=512, out_features=256)
        self.bn7 = mxnet.gluon.nn.BatchNorm1d(num_features=256, momentum=0.9)
        self.relu7 = mxnet.gluon.nn.ReLU()
        self.linear3 = mxnet.gluon.nn.Linear(in_features=256, out_features=10)

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
    x = mxnet.randn([2, 3, 2048]).to('cuda')
    y = model(x)
    return model

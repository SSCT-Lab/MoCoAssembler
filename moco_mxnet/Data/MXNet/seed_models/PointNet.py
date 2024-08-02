import mxnet
import mxnet as mx
from mxnet.gluon import nn


class PointNet(nn.Block):
    def __init__(self):
        super(PointNet, self).__init__()
        with self.name_scope():
            self.conv1 = mxnet.gluon.nn.Conv1D(channels=64, kernel_size=3, activation='relu')
            self.conv2 = mxnet.gluon.nn.Conv1D(channels=128, kernel_size=1, activation='relu')
            self.conv3 = mxnet.gluon.nn.Conv1D(channels=1024, kernel_size=1, activation='relu')
            self.fc1 = mxnet.gluon.nn.Dense(units=512, activation='relu')
            self.fc2 = mxnet.gluon.nn.Dense(units=256, activation='relu')
            self.fc3 = mxnet.gluon.nn.Dense(units=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def go():
    model = PointNet()
    model.initialize(mx.init.Xavier(), ctx=mx.cpu())
    x = mx.nd.random.uniform(shape=(3, 3, 100), ctx=mx.cpu())  
    y = model(x)
    return y

go()
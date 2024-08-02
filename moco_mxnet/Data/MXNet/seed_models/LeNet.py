import mxnet
import mxnet.gluon.nn as nn
import mxnet as mx


class LeNet(nn.Block):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = mxnet.gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu')
        self.pool1 = mxnet.gluon.nn.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = mxnet.gluon.nn.Conv2D(channels=16, kernel_size=5, activation='relu')
        self.pool2 = mxnet.gluon.nn.MaxPool2D(pool_size=2, strides=2)
        self.fc1 = mxnet.gluon.nn.Dense(units=120, activation='relu')
        self.fc2 = mxnet.gluon.nn.Dense(units=84, activation='relu')
        self.fc3 = mxnet.gluon.nn.Dense(units=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def go():
    model = LeNet()
    model.initialize(mx.init.Xavier(), ctx=mx.cpu())
    x = mx.nd.random.uniform(shape=(1, 1, 32, 32), ctx=mx.cpu())
    y = model(x)
    return model

go()
import mxnet
import mxnet as mx
from mxnet.gluon import nn


class AlexNet(nn.Block):
    def __init__(self):
        super(AlexNet, self).__init__()
        with self.name_scope():
            self.conv1 = mxnet.gluon.nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu')
            self.pool1 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
            self.conv2 = mxnet.gluon.nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu')
            self.pool2 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
            self.conv3 = mxnet.gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu')
            self.conv4 = mxnet.gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu')
            self.conv5 = mxnet.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
            self.pool3 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
            self.fc1 = mxnet.gluon.nn.Dense(units=4096, activation='relu')
            self.fc2 = mxnet.gluon.nn.Dense(units=4096, activation='relu')
            self.fc3 = mxnet.gluon.nn.Dense(units=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def go():
    model = AlexNet()
    model.initialize(mx.init.Xavier(), ctx=mx.cpu())
    x = mx.nd.random.uniform(shape=(3, 3, 227, 227), ctx=mx.cpu())
    y = model(x)
    return y

go()
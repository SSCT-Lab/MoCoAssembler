import mxnet
import mxnet as mx
import mxnet.gluon.nn as nn


class SqueezeNet(nn.Block):
    def __init__(self):
        super().__init__()
        self.relu2a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu2b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu2c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu3a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu3b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu3c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu4a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu4b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu4c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu5a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu5b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu5c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu6a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu6b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu6c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu7a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu7b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu7c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu8a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu8b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu8c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu9a = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu9b = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.relu9c = mxnet.gluon.nn.LeakyReLU(alpha=0.01)
        self.pool1 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.pool4 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.pool8 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.pool9 = mxnet.gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.avgpool = mxnet.gluon.nn.AvgPool2D(pool_size=3)

        self.conv1 = mxnet.gluon.nn.Conv2D(channels=96, kernel_size=7, strides=2)

        self.conv2a = mxnet.gluon.nn.Conv2D(channels=16, kernel_size=1, strides=1)
        self.conv2b = mxnet.gluon.nn.Conv2D(channels=64, kernel_size=1, strides=1)
        self.conv2c = mxnet.gluon.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1)

        self.conv3 = mxnet.gluon.nn.Conv2D(channels=16, kernel_size=1, strides=1)

        self.conv4a = mxnet.gluon.nn.Conv2D(channels=32, kernel_size=1, strides=1)
        self.conv4b = mxnet.gluon.nn.Conv2D(channels=128, kernel_size=1, strides=1)
        self.conv4c = mxnet.gluon.nn.Conv2D(channels=128, kernel_size=3, strides=1, padding=1)

        self.conv5 = mxnet.gluon.nn.Conv2D(channels=32, kernel_size=1, strides=1)

        self.conv6a = mxnet.gluon.nn.Conv2D(channels=48, kernel_size=1, strides=1)
        self.conv6b = mxnet.gluon.nn.Conv2D(channels=192, kernel_size=1, strides=1)
        self.conv6c = mxnet.gluon.nn.Conv2D(channels=192, kernel_size=3, strides=1, padding=1)

        self.conv7 = mxnet.gluon.nn.Conv2D(channels=48, kernel_size=1, strides=1)

        self.conv8a = mxnet.gluon.nn.Conv2D(channels=64, kernel_size=1, strides=1)
        self.conv8b = mxnet.gluon.nn.Conv2D(channels=256, kernel_size=1, strides=1)
        self.conv8c = mxnet.gluon.nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1)

        self.conv9 = mxnet.gluon.nn.Conv2D(channels=64, kernel_size=1, strides=1)

        self.conv10 = mxnet.gluon.nn.Conv2D(channels=512, kernel_size=1, strides=1)

        self.cat2 = mxnet.nd.concat
        self.cat3 = mxnet.nd.concat
        self.cat4 = mxnet.nd.concat
        self.cat5 = mxnet.nd.concat
        self.cat6 = mxnet.nd.concat
        self.cat7 = mxnet.nd.concat
        self.cat8 = mxnet.nd.concat
        self.cat9 = mxnet.nd.concat

        self.flatten = mxnet.gluon.nn.Flatten()
        self.fc = mxnet.gluon.nn.Dense(units=1000)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        y2 = self.conv2c(x)
        y2 = self.relu2c(y2)

        # 3rd block
        x = self.conv3(x)
        x = self.relu3a(x)
        x = self.conv2b(x)
        x = self.relu3b(x)
        y2 = self.conv2c(x)
        y2 = self.relu3c(y2)

        # 4th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)
        y2 = self.conv4c(x)
        y2 = self.relu4c(y2)
        x = self.pool4(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu5a(x)
        x = self.conv4b(x)
        x = self.relu5b(x)
        y2 = self.conv4c(x)
        y2 = self.relu5c(y2)

        # 6th block
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.conv6b(x)
        x = self.relu6b(x)
        y2 = self.conv6c(x)
        y2 = self.relu6c(y2)

        # 7th block
        x = self.conv7(x)
        x = self.relu7a(x)
        x = self.conv6b(x)
        x = self.relu7b(x)
        y2 = self.conv6c(x)
        y2 = self.relu7c(y2)

        # 8th block
        x = self.conv8a(x)
        x = self.relu8a(x)
        x = self.conv8b(x)
        x = self.relu8b(x)
        y2 = self.conv8c(x)
        y2 = self.relu8c(y2)
        x = self.pool8(x)

        # 9th block
        x = self.conv9(x)
        x = self.relu9a(x)
        x = self.conv8b(x)
        x = self.relu9b(x)
        y2 = self.conv8c(x)
        y2 = self.relu9c(y2)
        x = self.pool9(x)

        # 10th block
        x = self.conv10(x)

        # output
        x = self.flatten(x)
        x = self.fc(x)

        return x


def go():
    model = SqueezeNet()
    model.initialize(mx.init.Xavier(), ctx=mx.cpu())
    x = mx.nd.random.uniform(shape=(1, 3, 224, 224), ctx=mx.cpu())
    y = model(x)
    return model


go()

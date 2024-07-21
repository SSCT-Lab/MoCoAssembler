import mxnet
import mxnet.gluon.nn as nn


class squeezenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu2a = mxnet.gluon.nn.ReLU()
        self.relu2b = mxnet.gluon.nn.ReLU()
        self.relu2c = mxnet.gluon.nn.ReLU()
        self.relu3a = mxnet.gluon.nn.ReLU()
        self.relu3b = mxnet.gluon.nn.ReLU()
        self.relu3c = mxnet.gluon.nn.ReLU()
        self.relu4a = mxnet.gluon.nn.ReLU()
        self.relu4b = mxnet.gluon.nn.ReLU()
        self.relu4c = mxnet.gluon.nn.ReLU()
        self.relu5a = mxnet.gluon.nn.ReLU()
        self.relu5b = mxnet.gluon.nn.ReLU()
        self.relu5c = mxnet.gluon.nn.ReLU()
        self.relu6a = mxnet.gluon.nn.ReLU()
        self.relu6b = mxnet.gluon.nn.ReLU()
        self.relu6c = mxnet.gluon.nn.ReLU()
        self.relu7a = mxnet.gluon.nn.ReLU()
        self.relu7b = mxnet.gluon.nn.ReLU()
        self.relu7c = mxnet.gluon.nn.ReLU()
        self.relu8a = mxnet.gluon.nn.ReLU()
        self.relu8b = mxnet.gluon.nn.ReLU()
        self.relu8c = mxnet.gluon.nn.ReLU()
        self.relu9a = mxnet.gluon.nn.ReLU()
        self.relu9b = mxnet.gluon.nn.ReLU()
        self.relu9c = mxnet.gluon.nn.ReLU()
        self.pool1 = mxnet.gluon.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool4 = mxnet.gluon.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool8 = mxnet.gluon.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool9 = mxnet.gluon.nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = mxnet.gluon.nn.AdaptiveAvgPool2d(output_size=1)
        self.softmax = mxnet.gluon.nn.Softmax(dim=1)

        self.conv1 = mxnet.gluon.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)

        self.conv2a = mxnet.gluon.nn.Conv2d(in_channels=96, out_channels=16, kernel_size=1, stride=1)
        self.conv2b = mxnet.gluon.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1)
        self.conv2c = mxnet.gluon.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = mxnet.gluon.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)

        self.conv4a = mxnet.gluon.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.conv4b = mxnet.gluon.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.conv4c = mxnet.gluon.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = mxnet.gluon.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)

        self.conv6a = mxnet.gluon.nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1, stride=1)
        self.conv6b = mxnet.gluon.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=1, stride=1)
        self.conv6c = mxnet.gluon.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.conv7 = mxnet.gluon.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=1, stride=1)

        self.conv8a = mxnet.gluon.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1)
        self.conv8b = mxnet.gluon.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1)
        self.conv8c = mxnet.gluon.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv9 = mxnet.gluon.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)

        self.conv10 = mxnet.gluon.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)

        self.cat2 = mxnet.cat
        self.cat3 = mxnet.cat
        self.cat4 = mxnet.cat
        self.cat5 = mxnet.cat
        self.cat6 = mxnet.cat
        self.cat7 = mxnet.cat
        self.cat8 = mxnet.cat
        self.cat9 = mxnet.cat

        self.flatten = mxnet.gluon.nn.Flatten()
        self.fc = mxnet.gluon.nn.Linear(in_features=512, out_features=1000)

    def forward(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        y1 = self.conv2b(x)
        y1 = self.relu2b(y1)
        y2 = self.conv2c(x)
        y2 = self.relu2c(y2)
        x = self.cat2([y1, y2], dim=-1)

        # 3rd block
        x = self.conv3(x)
        x = self.relu3a(x)
        y1 = self.conv2b(x)
        y1 = self.relu3b(y1)
        y2 = self.conv2c(x)
        y2 = self.relu3c(y2)
        x = self.cat3([y1, y2], dim=-1)

        # 4th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        y1 = self.conv4b(x)
        y1 = self.relu4b(y1)
        y2 = self.conv4c(x)
        y2 = self.relu4c(y2)
        x = self.cat4([y1, y2], dim=-1)
        x = self.pool4(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu5a(x)
        y1 = self.conv4b(x)
        y1 = self.relu5b(y1)
        y2 = self.conv4c(x)
        y2 = self.relu5c(y2)
        x = self.cat5([y1, y2], dim=-1)

        # 6th block
        x = self.conv6a(x)
        x = self.relu6a(x)
        y1 = self.conv6b(x)
        y1 = self.relu6b(y1)
        y2 = self.conv6c(x)
        y2 = self.relu6c(y2)
        x = self.cat6([y1, y2], dim=-1)

        # 7th block
        x = self.conv7(x)
        x = self.relu7a(x)
        y1 = self.conv6b(x)
        y1 = self.relu7b(y1)
        y2 = self.conv6c(x)
        y2 = self.relu7c(y2)
        x = self.cat7([y1, y2], dim=-1)

        # 8th block
        x = self.conv8a(x)
        x = self.relu8a(x)
        y1 = self.conv8b(x)
        y1 = self.relu8b(y1)
        y2 = self.conv8c(x)
        y2 = self.relu8c(y2)
        x = self.cat8([y1, y2], dim=-1)
        x = self.pool8(x)

        # 9th block
        x = self.conv9(x)
        x = self.relu9a(x)
        y1 = self.conv8b(x)
        y1 = self.relu9b(y1)
        y2 = self.conv8c(x)
        y2 = self.relu9c(y2)
        x = self.cat9([y1, y2], dim=-1)
        x = self.pool9(x)

        # 10th block
        x = self.conv10(x)
        x = self.avgpool(x)

        # output
        x = self.softmax(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


def go():
    model = squeezenet()
    x = mxnet.randn((3, 3, 244, 244))
    y = model(x)
    return model

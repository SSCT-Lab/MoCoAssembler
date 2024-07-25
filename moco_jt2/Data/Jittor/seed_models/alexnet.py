import jittor
import jittor.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()

        self.relu1 = jittor.nn.ReLU()
        self.relu2 = jittor.nn.ReLU()
        self.relu3 = jittor.nn.ReLU()
        self.relu4 = jittor.nn.ReLU()
        self.relu5 = jittor.nn.ReLU()
        self.relu6 = jittor.nn.ReLU()
        self.relu7 = jittor.nn.ReLU()

        self.pool1 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = jittor.nn.AdaptiveAvgPool2d(output_size=6)

        self.conv1 = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = jittor.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = jittor.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = jittor.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = jittor.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.dropout1 = jittor.nn.Dropout(p=0.5)
        self.dropout2 = jittor.nn.Dropout(p=0.5)

        self.linear1 = jittor.nn.Linear(in_features=9216, out_features=4096)
        self.linear2 = jittor.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = jittor.nn.Linear(in_features=4096, out_features=1000)

        self.flatten = jittor.nn.Flatten()

    def execute(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)

        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)

        x = self.linear3(x)

        return x


def go():
    model = AlexNet()
    x = jittor.randn((1, 3, 224, 224))
    y = model(x)
    return model

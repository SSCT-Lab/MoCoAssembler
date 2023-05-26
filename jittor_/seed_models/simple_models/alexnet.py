import jittor
import jittor as jt
import jittor.nn as nn
from jittorsummary import summary


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

        self.conv1 = jittor.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = jittor.nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = jittor.nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = jittor.nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = jittor.nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.dropout1 = jittor.nn.Dropout(p=dropout)
        self.dropout2 = jittor.nn.Dropout(p=dropout)

        self.linear1 = jittor.nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.linear2 = jittor.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = jittor.nn.Linear(in_features=4096, out_features=num_classes)

    def execute(self, x):
        # 1st block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 3rd block
        x = self.conv3(x)
        x = self.relu3(x)

        # 4th block
        x = self.conv4(x)
        x = self.relu4(x)

        # 5th block
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        # 6th block
        x = self.avgpool(x)
        x = jt.flatten(x, 1)

        # 7th block
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)

        # 8th block
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)

        # output
        x = self.linear3(x)

        return x


if __name__ == '__main__':
    model = AlexNet()
    summary(model, (3, 224, 224))
    # x = jittor.randn((6, 3, 224, 224))
    # y = model(x)

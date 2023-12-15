import moco_jt
import moco_jt.nn as nn

class VGG_19(nn.Module):
    def __init__(self, class_num=1000):
        super().__init__()

        self.relu1a = moco_jt.nn.ReLU()
        self.relu1b = moco_jt.nn.ReLU()
        self.relu2a = moco_jt.nn.ReLU()
        self.relu2b = moco_jt.nn.ReLU()
        self.relu3a = moco_jt.nn.ReLU()
        self.relu3b = moco_jt.nn.ReLU()
        self.relu3c = moco_jt.nn.ReLU()
        self.relu3d = moco_jt.nn.ReLU()
        self.relu4a = moco_jt.nn.ReLU()
        self.relu4b = moco_jt.nn.ReLU()
        self.relu4c = moco_jt.nn.ReLU()
        self.relu4d = moco_jt.nn.ReLU()
        self.relu5a = moco_jt.nn.ReLU()
        self.relu5b = moco_jt.nn.ReLU()
        self.relu5c = moco_jt.nn.ReLU()
        self.relu5d = moco_jt.nn.ReLU()
        self.relu6 = moco_jt.nn.ReLU()
        self.relu7 = moco_jt.nn.ReLU()

        self.conv1a = moco_jt.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = moco_jt.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = moco_jt.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2b = moco_jt.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = moco_jt.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3b = moco_jt.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3c = moco_jt.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3d = moco_jt.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = moco_jt.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4b = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4c = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4d = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5a = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5b = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5c = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5d = moco_jt.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = moco_jt.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = moco_jt.nn.Linear(in_features=25088, out_features=4096)
        self.fc7 = moco_jt.nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = moco_jt.nn.Linear(in_features=4096, out_features=1000)

        self.softmax = moco_jt.nn.Softmax(dim=1)

    def execute(self, x):
        # 1st block
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)
        x = self.pool1(x)

        # 2nd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)
        x = self.pool2(x)

        # 3rd block
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)
        x = self.conv3c(x)
        x = self.relu3c(x)
        x = self.conv3d(x)
        x = self.relu3d(x)
        x = self.pool3(x)

        # 4th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)
        x = self.conv4c(x)
        x = self.relu4c(x)
        x = self.conv4d(x)
        x = self.relu4d(x)
        x = self.pool4(x)

        # 5th block
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)
        x = self.conv5c(x)
        x = self.relu5c(x)
        x = self.conv5d(x)
        x = self.relu5d(x)
        x = self.pool5(x)

        x = moco_jt.reshape(x, (-1, 512 * 7 * 7))

        # full connection
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.softmax(x)
        return x


def go():
    net = VGG_19()
    x = moco_jt.randn((1, 3, 224, 224))
    output = net(x)
    return net

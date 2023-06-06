import copy

import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = jittor.nn.Conv(1, 6, 5)
        self.conv_2 = jittor.nn.Sigmoid()
        self.conv_3 = jittor.nn.MaxPool2d(2, 2)

        self.conv_4 = jittor.nn.Conv(6, 16, 5)
        self.conv_5 = jittor.nn.Sigmoid()
        self.conv_6 = jittor.nn.MaxPool2d(2, 2)

        self.fc_1 = jittor.nn.Linear(400, 120)
        self.fc_2 = jittor.nn.Sigmoid()

        self.fc_3 = jittor.nn.Linear(120, 84)
        self.fc_4 = jittor.nn.Sigmoid()

        self.fc_5 = jittor.nn.Linear(84, 10)

    def execute(self, img):
        x = copy.deepcopy(img)
        # 1st block
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        # 2nd block
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        # 3rd block
        x = x.view(img.shape[0], -1)
        x = self.fc_1(x)
        x = self.fc_2(x)

        # 4th block
        x = self.fc_3(x)
        x = self.fc_4(x)
        x = self.fc_5(x)

        return x


# if __name__ == '__main__':
net = LeNet()
print(net(jittor.randn((224, 1, 32, 32))))

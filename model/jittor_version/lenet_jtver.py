import jittor as jt
import jittor.nn as nn
import jittorsummary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def execute(self, image):
        # x: batch-size*1@32*32
        # 1st block
        x = self.conv1(image)
        x = self.sigmoid(x)
        x = self.pool(x)

        # 2nd block
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        # 3rd block
        # x = jt.reshape(x, [image.shape[0], 1, -1])
        x = x.view(image.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        # 4th block
        x = self.fc2(x)
        x = self.sigmoid(x)

        # 5th block
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = LeNet()
    print(model)
    x = jt.randn((32, 1, 32, 32))
    y = model(x)
    print(y)

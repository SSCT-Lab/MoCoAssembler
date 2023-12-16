import jittor
import jittor.nn as nn


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.layer1 = jittor.nn.Sigmoid()
        self.layer2 = jittor.nn.Leaky_relu()
        self.layer3 = jittor.nn.AvgPool2d(kernel_size=299)
        self.layer4 = jittor.nn.Linear(in_features=1024, out_features=256)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def go():
    model = alexnet()
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
    return model

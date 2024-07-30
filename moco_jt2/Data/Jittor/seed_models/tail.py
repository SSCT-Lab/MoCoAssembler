import jittor
import jittor.nn as nn


class tail(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = jittor.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.flatten = jittor.nn.Flatten()
        self.tail_fc_10 = jittor.nn.Linear(in_features=1, out_features=10)
        self.tail_fc_1000 = jittor.nn.Linear(in_features=1, out_features=1000)

    def execute(self, x):
        x = self.conv1a(x)
        x = self.flatten(x)
        x = self.tail_fc_10(x)
        x = self.tail_fc_1000(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 3, 224, 224])
    m = tail()
    y = m(x)
    return m

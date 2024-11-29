import os

os.environ["disable_lock"] = "1"
import jittor
import jittor.nn as nn


class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jittor.nn.ConvTranspose3d(in_channels=1, kernel_size=5, out_channels=6, groups=9)

    def execute(self, x):
        x = self.conv1(x)
        return x


def go():
    jittor.flags.use_cuda = 1
    x = jittor.randn([1, 1, 28, 28])
    m = lenet()
    y = m(x)
    return list(y.shape)


go()
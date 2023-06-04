import jittor
import jittor.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()

        # api name mutate - conv
        self.conv1 = jittor.nn.Conv(in_channels = 6,out_channels = 8,kernel_size = 6)
    def execute(self, x):
        # shape fix
        x = x.reshape(-1, 6, 4, 4)
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    net = testnet()
    x = jittor.randn(3, 3, 32, 32)
    y = net(x)
    print(y)

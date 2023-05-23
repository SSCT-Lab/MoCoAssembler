import copy

import jittor
import jittor.nn as nn
import jittorsummary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
    def execute(self, img):
        x = copy.deepcopy(img)
        # 1st block
        return x


# if __name__ == '__main__':
net = LeNet()
print(net(jittor.randn((1, 1, 32, 32))))

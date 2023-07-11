import torch
import torch.nn as nn
import math
import numpy as np

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=0,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.block1 = BlockA(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2 = BlockB(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3 = BlockB(256,728,2,2,start_with_relu=True,grow_first=True)
        self.block4 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11 = BlockC(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12 = BlockD(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=1536)
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=2048)
        self.fc = torch.nn.Linear(in_features=2048,out_features=1000)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # # modified:----------------
                # mean = 0
                # std = math.sqrt(2. / n)
                # d = m.weight.data
                # d_n = (d - mean) / std
                # m.weight = torch.tensor(d_n)
                # # -------------------------
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # # modified---------------------
                # weight_ones = np.ones_like(m.weight.data)
                # m.weight = torch.tensor(weight_ones)
                # bias_zeros = np.zeros_like(m.bias.data)
                # m.bias = torch.tensor(bias_zeros)
                # # -----------------------------
        # -----------------------------
    def forward(self, x):

        # 1st block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 2nd block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 3rd block
        x = self.block1(x)

        # 4th block
        x = self.block2(x)

        # 5th block
        x = self.block3(x)

        # 6th block
        x = self.block4(x)

        # 7th block
        x = self.block5(x)

        # 8th block
        x = self.block6(x)

        # 9th block
        x = self.block7(x)

        # 10th block
        x = self.block8(x)

        # 11th block
        x = self.block9(x)

        # 12th block
        x = self.block10(x)

        # 13th block
        x = self.block11(x)

        # 14th block
        x = self.block12(x)

        # 15th block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 16th block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # x = F.adaptive_avg_pool2d(x, (1, 1))  modified
        aap = nn.AdaptiveAvgPool2d((1, 1))
        x = aap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = torch.nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class BlockA(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(BlockA, self).__init__()
        self.skip = torch.nn.Conv2d(in_filters,out_filters,1,stride=strides,bias=False)
        self.skipbn = torch.nn.BatchNorm2d(out_filters)
        self.relu = torch.nn.ReLU()
        self.rep = nn.Sequential(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.ReLU(),SeparableConv2d(out_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.MaxPool2d(3,strides,1))
    def forward(self, x):
        inp = x
        x = self.rep(x)
        skip = self.skip(inp)
        skip = self.skipbn(skip)
        x += skip
        return x
class BlockB(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(BlockB, self).__init__()
        self.skip = torch.nn.Conv2d(in_filters,out_filters,1,stride=strides,bias=False)
        self.skipbn = torch.nn.BatchNorm2d(out_filters)
        self.relu = torch.nn.ReLU()
        self.rep = nn.Sequential(torch.nn.ReLU(),SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.ReLU(),SeparableConv2d(out_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.MaxPool2d(3,strides,1))
    def forward(self, x):
        inp = x
        x = self.rep(x)
        skip = self.skip(inp)
        skip = self.skipbn(skip)
        x += skip
        return x
class BlockC(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(BlockC, self).__init__()
        self.relu = torch.nn.ReLU()
        self.rep = nn.Sequential(torch.nn.ReLU(),SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.ReLU(),SeparableConv2d(out_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.ReLU(),SeparableConv2d(out_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters))
    def forward(self, x):
        inp = x
        x = self.rep(x)
        x += inp
        return x
class BlockD(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(BlockD, self).__init__()
        self.skip = torch.nn.Conv2d(in_filters,out_filters,1,stride=strides,bias=False)
        self.skipbn = torch.nn.BatchNorm2d(out_filters)
        self.relu = torch.nn.ReLU()
        self.rep = nn.Sequential(torch.nn.ReLU(),SeparableConv2d(in_filters,in_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(in_filters),torch.nn.ReLU(),SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False),torch.nn.BatchNorm2d(out_filters),torch.nn.MaxPool2d(3,strides,1))
    def forward(self, x):
        inp = x
        x = self.rep(x)
        skip = self.skip(inp)
        skip = self.skipbn(skip)
        x += skip
        return x
def go():
    device = torch.device('cuda')
    model = Xception().to(device)
    x = torch.randn(3, 3, 224, 224).to(device)
    y = model(x)
    return model

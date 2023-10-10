import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *



class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, spatial, extent, extra_params, mlp,
                 dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, spatial, extent, extra_params,
                                      mlp, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, spatial, extent, extra_params, mlp,
                    dropRate):
        layers = []
        for i in range(int(nb_layers)):
            if i == 0 :
                layers.append(block(in_planes, out_planes, stride, spatial, extent,
                                    extra_params, mlp,
                                    dropRate))
            else:
                layers.append(block(out_planes, out_planes, 1, spatial, extent,
                                    extra_params, mlp,
                                    dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class GENet(nn.Module):
    def __init__(self, num_classes=1000, extent=0, extra_params=True, mlp=True, dropRate=0.0):
        super(GENet, self).__init__()

        layer_nums = [3, 4, 6, 3]
        in_channels = [64, 256, 512, 1024]
        out_channels = [256, 512, 1024, 2048]
        self.out_channels = out_channels
        strides = [1, 2, 2, 2]
        spatial = [56, 28, 14, 7]

        block = GEBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Manually entering the input size for the global depthwise convolution to work. This is not ideal.

        # 1st block
        self.block1 = NetworkBlock(layer_nums[0], in_channels[0], out_channels[0], block, 1, spatial=spatial[0], extent = extent,
                                   extra_params=extra_params, mlp=mlp, dropRate=dropRate)
        # 2nd block
        self.block2 = NetworkBlock(layer_nums[1], in_channels[1], out_channels[1], block, 2, spatial=spatial[1], extent = extent,
                                   extra_params=extra_params, mlp=mlp, dropRate=dropRate)
        # 3rd block
        self.block3 = NetworkBlock(layer_nums[2], in_channels[2], out_channels[2], block, 2, spatial=spatial[2], extent = extent,
                                   extra_params=extra_params, mlp=mlp, dropRate=dropRate)

        self.block4 = NetworkBlock(layer_nums[3], in_channels[3], out_channels[3], block, 2, spatial=spatial[3], extent = extent,
                                   extra_params=extra_params, mlp=mlp, dropRate=dropRate)

        # global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avg_pool(out)
        out = out.view(-1, self.out_channels[3])
        return self.fc(out)

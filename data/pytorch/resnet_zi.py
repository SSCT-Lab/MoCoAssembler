import torch
from .modules.fixup import ZIConv2d


__all__ = ['resnet_zi']


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, pre_bias=True, post_bias=True, multiplier=False):
    return ZIConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, groups=groups, bias=bias, pre_bias=pre_bias, post_bias=post_bias, multiplier=multiplier)


class BasicBlock(torch.nn.Module):
    def __init__(self, inplanes, planes,  stride=1, expansion=1, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes,
                             groups=groups, multiplier=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, layer_depth=1):
        super(Bottleneck, self).__init__()

        self.conv1 = ZIConv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.conv3 = ZIConv2d(
            planes, planes * expansion, kernel_size=1, multiplier=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.layer_depth = layer_depth

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNetZI(torch.nn.Module):
    def __init__(self):
        super(ResNetZI, self).__init__()
        self.num_layers = 1

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = ZIConv2d(self.inplanes, out_planes,
                                  kernel_size=1, stride=stride, bias=False)
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, layer_depth=self.num_layers))
        self.inplanes = planes * expansion
        self.num_layers += 1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, layer_depth=self.num_layers))
            self.num_layers += 1

        return torch.nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

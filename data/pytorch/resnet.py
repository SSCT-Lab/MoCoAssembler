import torch
from utils.mixup import MixUp

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, groups=groups, bias=bias)


class BasicBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(BasicBlock, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups)
        self.bn2 = torch.nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None, groups=1, residual_block=None,
                 dropout=0.):
        super(Bottleneck, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(
            planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None,
                    mixup=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, out_planes,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, dropout=dropout))
        if mixup:
            layers.append(MixUp())
        return torch.nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
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

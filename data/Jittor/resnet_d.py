import jittor as jt
from jittor import init
from jittor import nn

__all__ = ['ResNet', 'resnet50', 'resnet101']
model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv(
            planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv(planes, (planes * self.expansion), 1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes * self.expansion))
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(3, 64, stride=2), nn.BatchNorm2d(
            64), nn.ReLU(), conv3x3(64, 64), nn.BatchNorm2d(64), nn.ReLU(), conv3x3(64, 128))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Pool(7, stride=1, op='mean')
        self.fc = nn.Linear((512 * block.expansion), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, value=1)
                init.constant_(m.bias, value=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(nn.Conv(self.inplanes, (planes * block.expansion),
                                       1, stride=stride, bias=False), nn.BatchNorm2d((planes * block.expansion)))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = (planes * block.expansion)
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view((x.shape[0], (- 1)))
        x = self.fc(x)
        return x


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_parameters(
            jt.load('./pretrained_models/resnet50-deep.pth'))
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_parameters(
            jt.load('./pretrained_models/resnet101-deep.pth'))
    return model
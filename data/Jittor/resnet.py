from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
Borrow Codes From : 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.utils.model_zoo as model_zoo
import math
import os
from jittor import nn, Module, init
import jittor as jt


# __all__ = []

class AffineChannel2d(Module):
    """ A simple channel-wise affine transformation operation """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = jt.zeros((num_features))
        self.bias = jt.zeros((num_features))
        init.uniform_(self.weight, 0, 1)
        init.constant_(self.bias, 0.0)

    def execute(self, x):
        return x * self.weight.reshape(1, self.num_features, 1, 1) + \
            self.bias.reshape(1, self.num_features, 1, 1)


_BN = nn.BatchNorm
# _BN = AffineChannel2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def init_with_pretrain(model, pretrained_dict):
    params = model.parameters()
    model_dict = {}
    for p in params:
        model_dict[p.name()] = p.data
    n_model = len(model_dict)
    n_pretrain = len(pretrained_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    n_init = len(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_parameters(model_dict)
    print('total params in model is %d, in pretrained model is %d, init %d' % (n_model, n_pretrain, n_init))


class ResnetXtBottleneck(Module):
    '''
    Resnet50 == ResnetXt50(1*64d)
    conv2:
        ->[64], 56*56
        nn.Sequential(
            ResnetXtBottleneck(64, 256, 1, 1, True, 1, 64), -> [256], 56*56
            ResnetXtBottleneck(256, 256, 1, 1, False, 1, 64), -> [256], 56*56
            ResnetXtBottleneck(256, 256, 1, 1, False, 1, 64), -> [256], 56*56
        )
    conv3:
        ->[256], 56*56
        nn.Sequential(
            ResnetXtBottleneck(256, 512, 2, 2, True, 1, 64), -> [512], 28*28
            ResnetXtBottleneck(512, 512, 1, 2, False, 1, 64), -> [512], 28*28
            ResnetXtBottleneck(512, 512, 1, 2, False, 1, 64), -> [512], 28*28
        )
    '''

    def __init__(self, inplanes, outplanes, stride=1, widen_factor=1, has_shortcut=False, cardinality=1, base_width=64):
        super(ResnetXtBottleneck, self).__init__()

        D = cardinality * base_width * (2 ** (widen_factor - 1))

        self.conv1 = nn.Conv(inplanes, D, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = _BN(D)

        self.conv2 = nn.Conv(D, D, kernel_size=3, stride=1, padding=1, groups=cardinality, bias=False)
        self.bn2 = _BN(D)

        self.conv3 = nn.Conv(D, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = _BN(outplanes)

        self.has_shortcut = has_shortcut
        if has_shortcut:
            self.conv4 = nn.Conv(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn4 = _BN(outplanes)

        self.relu = nn.ReLU()

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.has_shortcut:
            residual = self.bn4(self.conv4(x))

        out += residual
        out = self.relu(out)

        return out


class ResnetXtFPN(Module):

    def __init__(self, layers, cardinality=1, base_width=64, usefpn=False, num_classes=None):
        super(ResnetXtFPN, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.usefpn = usefpn
        self.num_classes = num_classes

        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _BN(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')

        self.layer1 = self._make_layer(inplanes=64, outplanes=256, stride=1, widen_factor=1, blocks=layers[0])
        self.layer2 = self._make_layer(inplanes=256, outplanes=512, stride=2, widen_factor=2, blocks=layers[1])
        self.layer3 = self._make_layer(inplanes=512, outplanes=1024, stride=2, widen_factor=3, blocks=layers[2])
        if len(layers) == 4:
            self.layer4 = self._make_layer(inplanes=1024, outplanes=2048, stride=2, widen_factor=4, blocks=layers[3])
        else:
            self.layer4 = None

        if usefpn:
            # c1, c2, c3, c4 all conv to same channel
            self.fpn_c4p4 = nn.Conv(2048, 256, kernel_size=1, stride=1, padding=0)
            self.fpn_c3p3 = nn.Conv(1024, 256, kernel_size=1, stride=1, padding=0)
            self.fpn_c2p2 = nn.Conv(512, 256, kernel_size=1, stride=1, padding=0)
            self.fpn_c1p1 = nn.Conv(256, 256, kernel_size=1, stride=1, padding=0)

            self.fpn_p4 = nn.Conv(256, 256, kernel_size=3, stride=1, padding=1)
            self.fpn_p3 = nn.Conv(256, 256, kernel_size=3, stride=1, padding=1)
            self.fpn_p2 = nn.Conv(256, 256, kernel_size=3, stride=1, padding=1)
            self.fpn_p1 = nn.Conv(256, 256, kernel_size=3, stride=1, padding=1)

        if (num_classes is not None) and (not usefpn):
            self.avgpool = nn.Pool(7, stride=1, op='mean')
            self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, _BN):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)

    def _make_layer(self, inplanes, outplanes, stride, widen_factor, blocks):
        layers = []
        layers.append(
            ResnetXtBottleneck(inplanes, outplanes, stride, widen_factor, True, self.cardinality, self.base_width))
        for i in range(1, blocks):
            layers.append(
                ResnetXtBottleneck(outplanes, outplanes, 1, widen_factor, False, self.cardinality, self.base_width))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return nn.upsample(x, size=(H, W), mode='bilinear', align_corners=True) + y  # 'bilinear'

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = x = self.maxpool(x)

        c1 = x = self.layer1(x)
        c2 = x = self.layer2(x)
        c3 = x = self.layer3(x)
        if self.layer4 is not None:
            c4 = x = self.layer4(x)

        if self.usefpn:
            # Top-down
            p4 = self.fpn_c4p4(c4)
            p3 = self._upsample_add(p4, self.fpn_c3p3(c3))
            p2 = self._upsample_add(p3, self.fpn_c2p2(c2))
            p1 = self._upsample_add(p2, self.fpn_c1p1(c1))

            # Attach 3x3 conv to all P layers to get the final feature maps.
            p4 = self.fpn_p4(p4)
            p3 = self.fpn_p3(p3)
            p2 = self.fpn_p2(p2)
            p1 = self.fpn_p1(p1)

            return [p1, p2, p3, p4]

        else:
            if self.num_classes is not None:
                x = self.avgpool(x)
                x = x.reshape(x.shape[0], -1)
                x = self.fc(x)
            return x


def convert_official(model, src_url, dst_file):
    pretrain_dict = model_zoo.load_url(src_url)
    params = model.parameters()
    model_dict = {}
    for p in params:
        model_dict[p.name()] = p.data
    for key, value in pretrain_dict.items():
        if 'downsample' in key:
            key = key.replace('downsample.0', 'conv4')
            key = key.replace('downsample.1', 'bn4')
        if key not in model_dict.keys():
            print('[imagenet pretrain] add', key, 'to the model_dict.')
        else:
            assert model_dict[key].shape == value.shape, 'shape of [%s] is not match!' % key
        model_dict[key] = value
    torch.save(model_dict, dst_file)


'''
resnet50(resnetXt50_1x64d), resnet101(resnetXt101_1x64d)
'''


def resnet50C4(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 6], cardinality=1, base_width=64, usefpn=False, num_classes=num_classes)
    if pretrained:
        pretrain_file = './imagenet_pretrain/resnet50_from_modelzoo.pth'
        if not os.path.exists(pretrain_file):
            os.makedirs("./imagenet_pretrain", exist_ok=True)
            convert_official(model, model_urls['resnet50'], pretrain_file)
        init_with_pretrain(model, torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    return model


def resnet50(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=1, base_width=64, usefpn=False, num_classes=num_classes)
    if pretrained:
        pretrain_file = './imagenet_pretrain/resnet50_from_modelzoo.pth'
        if not os.path.exists(pretrain_file):
            os.makedirs("./imagenet_pretrain", exist_ok=True)
            convert_official(model, model_urls['resnet50'], pretrain_file)
        init_with_pretrain(model, torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    return model


def resnet101(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=1, base_width=64, usefpn=False, num_classes=num_classes)
    if pretrained:
        pretrain_file = './imagenet_pretrain/resnet101_from_modelzoo.pth'
        if not os.path.exists(pretrain_file):
            os.makedirs("./imagenet_pretrain", exist_ok=True)
            convert_official(model, model_urls['resnet101'], pretrain_file)
        init_with_pretrain(model, torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    return model


'''
resnet50FPN, resnet101FPN
'''


def resnet50FPN(pretrained=False):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=1, base_width=64, usefpn=True)
    if pretrained:
        pretrain_file = './imagenet_pretrain/resnet50_from_modelzoo.pth'
        if not os.path.exists(pretrain_file):
            os.makedirs("./imagenet_pretrain", exist_ok=True)
            convert_official(model, model_urls['resnet50'], pretrain_file)
        init_with_pretrain(model, torch.load(pretrain_file))  # , map_location=lambda storage, loc: storage))
    return model


def resnet101FPN(pretrained=False):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=1, base_width=64, usefpn=True)
    if pretrained:
        pretrain_file = './imagenet_pretrain/resnet101_from_modelzoo.pth'
        if not os.path.exists(pretrain_file):
            os.makedirs("./imagenet_pretrain", exist_ok=True)
            convert_official(model, model_urls['resnet101'], pretrain_file)
        init_with_pretrain(model, torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    return model


'''
resnetXt50_32x4d, resnetXt101_32x4d, resnetXt101_64x4d, resnetXt101_64x4d
'''


def resnetXt50_32x4d(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=32, base_width=4, usefpn=False, num_classes=num_classes)
    if pretrained:
        pass
    return model


def resnetXt101_32x4d(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=32, base_width=4, usefpn=False, num_classes=num_classes)
    if pretrained:
        pass
    return model


def resnetXt50_64x4d(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=64, base_width=4, usefpn=False, num_classes=num_classes)
    if pretrained:
        pass
    return model


def resnetXt101_64x4d(pretrained=False, num_classes=None):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=64, base_width=4, usefpn=False, num_classes=num_classes)
    if pretrained:
        pass
    return model


'''
resnetXt50FPN_32x4d, resnetXt101FPN_32x4d, resnetXt101FPN_64x4d, resnetXt101FPN_64x4d
'''


def resnetXt50FPN_32x4d(pretrained=False):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=32, base_width=4, usefpn=True)
    if pretrained:
        pass
    return model


def resnetXt101FPN_32x4d(pretrained=False):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=32, base_width=4, usefpn=True)
    if pretrained:
        pass
    return model


def resnetXt50FPN_64x4d(pretrained=False):
    model = ResnetXtFPN([3, 4, 6, 3], cardinality=64, base_width=4, usefpn=True)
    if pretrained:
        pass
    return model


def resnetXt101FPN_64x4d(pretrained=False):
    model = ResnetXtFPN([3, 4, 23, 3], cardinality=64, base_width=4, usefpn=True)
    if pretrained:
        pass
    return model


if __name__ == '__main__':
    # backbone_net = resnet101(pretrained=True, num_classes=1000).cuda()
    backbone_net = resnet50C4(pretrained=True, num_classes=None)

    backbone_net.train()
    input = jt.random((1, 3, 224, 224))
    output = backbone_net(input)
    print(output.shape)
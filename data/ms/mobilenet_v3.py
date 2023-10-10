from typing import Optional, List

import mindspore.nn as nn
import mindspore.ops as ops

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.blocks import SqueezeExcite
from .layers.pooling import GlobalAvgPooling
from .utils import load_pretrained, make_divisible
from .registry import register_model
from mission.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

'''
1. 基本结构和timm/torchvision都是一样的，没有太大差别, 但是
网络最后的分类器pytorch是直接用全连接层做分类器，model zoo是参考论文的用法，直接用卷积层做分类器，
2. 同样将 conv2d+bn+relu 改成了Conv2dNormActivation
3. 全局平均池化使用提出来公用的GlobalAvgPooling
'''


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'dataset_transform': {
            'transforms_imagenet_train': {
                'image_resize': 224,
                'scale': (0.08, 1.0),
                'ratio': (0.75, 1.333),
                'hflip': 0.5,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
            'transforms_imagenet_eval': {
                'image_resize': 224,
                'crop_pct': DEFAULT_CROP_PCT,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
        },
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'mobilenet_v3_small_1.0': _cfg(url=''),
    'mobilenet_v3_large_1.0': _cfg(url=''),
    'mobilenet_v3_small_0.75': _cfg(url=''),
    'mobilenet_v3_large_0.75': _cfg(url='')
}


class ResUnit(nn.Cell):

    def __init__(self, num_in: int,
                 num_mid: int,
                 num_out: int,
                 kernel_size: int,
                 norm: nn.Cell,
                 activation: str,
                 stride: int = 1,
                 use_se: bool = False) -> None:
        super(ResUnit, self).__init__()
        self.use_se = use_se
        self.use_short_cut_conv = num_in == num_out and stride == 1
        self.use_hs = activation == 'hswish'
        self.activation = nn.HSwish if self.use_hs else nn.ReLU

        layers = []

        # Expand.
        if num_in != num_mid:
            layers.append(
                Conv2dNormActivation(num_in, num_mid, kernel_size=1, norm=norm, activation=self.activation)
            )

        # DepthWise.
        layers.append(
            Conv2dNormActivation(num_mid, num_mid, kernel_size=kernel_size, stride=stride, groups=num_mid, norm=norm,
                               activation=self.activation)
        )
        if use_se:
            squeeze_channel = make_divisible(num_mid // 4, 8)
            layers.append(
                SqueezeExcite(num_mid, squeeze_channel, nn.ReLU, nn.HSigmoid)
            )

        # Project.
        layers.append(
            Conv2dNormActivation(num_mid, num_out, kernel_size=1, norm=norm, activation=None)
        )

        self.block = nn.SequentialCell(layers)
        self.add = ops.Add()

    def construct(self, x):
        out = self.block(x)

        if self.use_short_cut_conv:
            out = self.add(out, x)

        return out


class MobileNetV3(nn.Cell):

    def __init__(self,
                 model_cfgs: List,
                 last_channel: int,
                 in_channels: int = 3,
                 multiplier: float = 1.0,
                 norm: Optional[nn.Cell] = None,
                 round_nearest: int = 8,
                 num_classes: int = 1000,
                 drop_rate: float = 0.2
                 ) -> None:
        super(MobileNetV3, self).__init__()

        if not norm:
            norm = nn.BatchNorm2d

        self.inplanes = 16
        layers = []

        # Building first layer.
        first_conv_in_channel = in_channels
        first_conv_out_channel = make_divisible(self.inplanes * multiplier, round_nearest)
        layers.append(
            Conv2dNormActivation(
                first_conv_in_channel,
                first_conv_out_channel,
                kernel_size=3,
                stride=2,
                norm=norm,
                activation=nn.HSwish
            )
        )

        # Building inverted residual blocks.
        for layer_cfg in model_cfgs:
            layers.append(self._make_layer(kernel_size=layer_cfg[0],
                                           exp_ch=make_divisible(multiplier * layer_cfg[1], round_nearest),
                                           out_channel=make_divisible(multiplier * layer_cfg[2], round_nearest),
                                           use_se=layer_cfg[3],
                                           activation=layer_cfg[4],
                                           stride=layer_cfg[5],
                                           norm=norm
                                           )
                          )

        lastconv_input_channel = make_divisible(multiplier * model_cfgs[-1][2], round_nearest)
        lastconv_output_channel = lastconv_input_channel * 6

        # Building last several layers.
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channel,
                lastconv_output_channel,
                kernel_size=1,
                norm=norm,
                activation=nn.HSwish
            )
        )

        self.features = nn.SequentialCell(layers)

        self.pool = GlobalAvgPooling(keep_dim=True)
        self.num_features = last_channel
        self.conv_head = nn.SequentialCell([
            nn.Conv2d(in_channels=lastconv_output_channel,
                      out_channels=self.num_features,
                      kernel_size=1,
                      stride=1),
            nn.HSwish(),
            nn.Dropout(keep_prob=1 - drop_rate),
        ])

        self.classifier = nn.Conv2d(in_channels=self.num_features,
                                    out_channels=num_classes,
                                    kernel_size=1,
                                    has_bias=True)
        self.squeeze = ops.Squeeze(axis=(2, 3))

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.classifier = nn.Conv2d(in_channels=self.num_features,
                                    out_channels=num_classes,
                                    kernel_size=1,
                                    has_bias=True)

    def get_features(self, x):
        x = self.features(x)
        return x

    def construct(self, x):
        x = self.get_features(x)
        x = self.pool(x)
        x = self.conv_head(x)
        x = self.classifier(x)
        x = self.squeeze(x)

        return x

    def _make_layer(self,
                    kernel_size: int,
                    exp_ch: int,
                    out_channel: int,
                    use_se: bool,
                    activation: str,
                    norm: nn.Cell,
                    stride: int = 1
                    ):
        """Block layers."""
        layer = ResUnit(self.inplanes, exp_ch, out_channel,
                        kernel_size=kernel_size, stride=stride, activation=activation, use_se=use_se, norm=norm)
        self.inplanes = out_channel

        return layer


model_cfgs = {
    "large": [
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', 2],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', 2],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hswish', 2],
        [3, 200, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 480, 112, True, 'hswish', 1],
        [3, 672, 112, True, 'hswish', 1],
        [5, 672, 160, True, 'hswish', 2],
        [5, 960, 160, True, 'hswish', 1],
        [5, 960, 160, True, 'hswish', 1]
    ],
    "small": [
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hswish', 2],
        [5, 240, 40, True, 'hswish', 1],
        [5, 240, 40, True, 'hswish', 1],
        [5, 120, 48, True, 'hswish', 1],
        [5, 144, 48, True, 'hswish', 1],
        [5, 288, 96, True, 'hswish', 2],
        [5, 576, 96, True, 'hswish', 1],
        [5, 576, 96, True, 'hswish', 1]]
}


@register_model
def mobilenet_v3_small_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v3_small_1.0']
    model = MobileNetV3(model_cfgs=model_cfgs['small'],
                        last_channel=1024,
                        in_channels=in_channels,
                        multiplier=1.0,
                        num_classes=num_classes,
                        **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_large_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v3_large_1.0']
    model = MobileNetV3(model_cfgs=model_cfgs['large'],
                        last_channel=1280,
                        in_channels=in_channels,
                        multiplier=1.0,
                        num_classes=num_classes,
                        **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_small_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v3_small_0.75']
    model = MobileNetV3(model_cfgs=model_cfgs['small'],
                        last_channel=1024,
                        in_channels=in_channels,
                        multiplier=0.75,
                        num_classes=num_classes,
                        **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_large_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v3_large_0.75']
    model = MobileNetV3(model_cfgs=model_cfgs['large'],
                        last_channel=1280,
                        in_channels=in_channels,
                        multiplier=0.75,
                        num_classes=num_classes,
                        **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

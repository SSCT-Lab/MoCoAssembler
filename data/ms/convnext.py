# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ConvNext Model Define"""

import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.common import initializer as weight_init

from typing import List
from registry import register_model
from mission.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


'''
1. 添加了DropPath2D正则化算子与Identity类;
2. 添加了ImageNet数据集的预处理配置_cfg(RandomColorAdjust与HWC2CHW定义在公用接口transforms_factory.py中);
3. torchvision的ConvNextLayerNorm实现中，只有norm_axis = 1的情况;
4. 对于训练时随机丢弃网络节点的拓扑结构变化操作，MindSpore使用的DropPath算子，而torch使用的是随机深度算子;
5. torch实现时单独传入了一个CNBlockConfig类来构建ConvNext网络，而MindSpore直接传入形参;
6. 搭建ConvNext网络时，torch顺序:stem->stage->downsample，而MindSpore顺序:stem->downsample->stage，最终都是全局池化和分类层;
7. 暂时未添加测试网络的main()函数;
'''


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000,
        'dataset_transform': {
            # args中没找到interpolation？
            'transforms_imagenet_train': {
                'image_resize': 224,
                'hflip': 0.5,
                'interpolation': 'bicubic',
                'mean': IMAGENET_DEFAULT_MEAN / 255,
                'std': IMAGENET_DEFAULT_STD / 255,
            },
            'transforms_imagenet_eval': {
                # int(256 / 224 * image_size)
                'image_resize': int(256 / 224 * 224),
                'crop_pct': DEFAULT_CROP_PCT,
                'interpolation': 'bicubic',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
        },
        **kwargs
    }


default_cfgs = {
    'convnext_tiny': _cfg(url=''),
    'convnext_small': _cfg(url=''),
    'convnext_base': _cfg(url=''),
    'convnext_large': _cfg(url=''),
    'convnext_xlarge': _cfg(url=''),
}


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x


# torch的dropout概率为drop_prob, 而MindSpore为1-drop_prob
class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self, drop_prob):
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)


# torchvision的实现中，只有norm_axis = 1的情况
class ConvNextLayerNorm(nn.LayerNorm):
    """ConvNextLayerNorm"""

    def __init__(self, normalized_shape, epsilon, norm_axis=-1):
        super(ConvNextLayerNorm, self).__init__(normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1, 1), "ConvNextLayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x):
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.Transpose()(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.Transpose()(y, (0, 3, 1, 2))
        return y


class Block(nn.Cell):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Dense -> GELU -> Dense; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        # MindSpore和torch都是用全连接层实现1*1卷积和逐点卷积
        self.block = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=7, group=dim, has_bias=True),  # depthwise conv
            ops.transpose((0, 2, 3, 1)),
            ConvNextLayerNorm((dim,), epsilon=1e-6),
            nn.Dense(dim, 4 * dim),  # pointwise/1x1 convs, implemented with Dense layers
            nn.GELU(),
            nn.Dense(4 * dim, dim)
        )
        self.gamma = Parameter(Tensor(layer_scale_init_value * np.ones(dim), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        # 对于训练时随机丢弃网络节点的拓扑结构变化操作，MindSpore使用的DropPath算子，而torch使用的是随机深度算子
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

    def construct(self, x):
        """Block construct"""
        downsample = x
        x = self.block(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.Transpose()(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


# torch实现时单独传入了一个CNBlockConfig类来构建ConvNext网络，而MindSpore直接传入形参
class ConvNeXt(nn.Cell):

    def __init__(self, in_chans, num_classes, depths,
                 dims: List[int],
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()

        self.downsample_layers = nn.CellList()  # stem and 3 intermediate down_sampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6)  # final norm layer
        self.head = nn.Dense(dims[-1], num_classes) # classifier

        self.init_weights()
        self.head.weight.set_data(self.head.weight * head_init_scale)
        self.head.bias.set_data(self.head.bias * head_init_scale)

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def convnext_tiny(**kwargs):
    """convnext_tiny"""
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
    return model


@register_model
def convnext_small(**kwargs):
    """convnext_small"""
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
    return model


@register_model
def convnext_base(**kwargs):
    """convnext_base"""
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     **kwargs)
    return model


@register_model
def convnext_large(**kwargs):
    """convnext_large"""
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     **kwargs)
    return model


@register_model
def convnext_xlarge(**kwargs):
    """convnext_xlarge"""
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     **kwargs)
    return model

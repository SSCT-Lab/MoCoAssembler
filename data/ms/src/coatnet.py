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
"""CoAtNet model definition"""


import numpy as np
import mindspore as ms
import mindspore.nn as nn


class DropPath(nn.Cell):
    """DropPath"""
    def __init__(self, survival_prob=1.0):
        super(DropPath, self).__init__()
        self.survival_prob = survival_prob

    def construct(self, x):
        if not self.training:
            return x
        if self.survival_prob == 1:
            return x
        x_shape = ms.ops.Shape()(x)
        # random_tensor = ms.ops.UniformReal()((x_shape[0], 1, 1, 1))
        random_tensor = ms.ops.UniformReal()((x_shape[0],) + (x.ndim - 1) * (1,))
        x = x / self.survival_prob * ms.ops.Floor()(random_tensor + self.survival_prob)
        return x


# class DropPath_(nn.Cell):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """

#     def __init__(self, survival_prob, ndim):
#         super(DropPath_, self).__init__()
#         self.survival_prob = survival_prob
#         self.ndim = ndim
#         shape = (1,) + (1,) * (ndim + 1)
#         self.mask = ms.Tensor(np.ones(shape), dtype=ms.dtype.float32)
#         self.drop = nn.Dropout(keep_prob=survival_prob)

#     def construct(self, x):
#         if not self.training:
#             return x
#         if self.survival_prob == 1:
#             return x
#         mask = ms.ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
#         x = self.drop(mask) * x
#         return x


# class DropPath2d(DropPath_):
#     def __init__(self, survival_prob):
#         super(DropPath2d, self).__init__(survival_prob=survival_prob, ndim=2)


class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid", data_format="NCHW"):
        super(MaxPool2d, self).__init__()
        self.ksizes = self._get_value(kernel_size)
        self.strides = self._get_value(stride)
        self.rates = [1, 1, 1, 1]
        self.data_format = data_format
        self.unfold = nn.Unfold(self.ksizes, self.strides, self.rates, pad_mode)

    def _get_value(self, value):
        if isinstance(value, int):
            return [1, value, value, 1]
        elif len(value) == 2:
            return [1, value[0], value[1], 1]
        elif len(value) == 4:
            return value
        else:
            raise ValueError

    def construct(self, x):
        if self.data_format == "NHWC":
            x = ms.ops.Transpose()(x, (0, 3, 1, 2))
        x_shape = ms.ops.Shape()(x)
        x = self.unfold(x)
        x = ms.ops.Reshape()(x,
                             (x_shape[0],
                              self.ksizes[1] * self.ksizes[2],
                              x_shape[1],
                              (x_shape[2] - (self.ksizes[1] + (self.ksizes[1] - 1) * (self.rates[1] - 1))) // self.strides[1] + 1,
                              (x_shape[3] - (self.ksizes[2] + (self.ksizes[2] - 1) * (self.rates[2] - 1))) // self.strides[2] + 1))
        x = ms.ops.ReduceMax(keep_dims=False)(x, 1)
        if self.data_format == "NHWC":
            x = ms.ops.Transpose()(x, (0, 2, 3, 1))
        return x


class Stem(nn.Cell):
    """Stem"""
    def __init__(self, input_filters, output_filters, bn_eps=0.001, bn_momentum=0.99):
        super(Stem, self).__init__()

        self.conv = nn.SequentialCell([
            nn.Conv2d(input_filters, output_filters, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm2d(output_filters, eps=bn_eps, momentum=bn_momentum),
            nn.GELU(),
            nn.Conv2d(output_filters, output_filters, kernel_size=3, stride=1, has_bias=False),
        ])

    def construct(self, x):
        x = self.conv(x)
        return x


class GlobalAvgPool2d(nn.Cell):
    """GlobalAvgPool2d"""
    def __init__(self, keep_dims=False, data_format="NCHW"):
        super(GlobalAvgPool2d, self).__init__()
        self.keep_dims = keep_dims
        self.data_format = data_format
        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError

    def construct(self, x):
        if self.data_format == "NCHW":
            x = ms.ops.ReduceMean(keep_dims=self.keep_dims)(x, (2, 3))
        else:
            x = ms.ops.ReduceMean(keep_dims=self.keep_dims)(x, (1, 2))
        return x


class Swish(nn.Cell):
    """Swish"""
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = x * self.sigmoid(x)
        return x


class SE(nn.Cell):
    """SE"""
    def __init__(self, input_filters, reduced_filters):
        super(SE, self).__init__()

        self.se = nn.SequentialCell([
            GlobalAvgPool2d(keep_dims=True),
            nn.Conv2d(input_filters, reduced_filters, kernel_size=1, has_bias=True),
            Swish(),
            nn.Conv2d(reduced_filters, input_filters, kernel_size=1, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        x = x * self.se(x)
        return x


class MBConv(nn.Cell):
    """Pre-activation MBConv"""
    def __init__(self, input_filters, output_filters, resolution, drop_path_rate, downsample=False,
                 expand_ratio=4, se_ratio=0.25, bn_eps=0.001, bn_momentum=0.99):
        super(MBConv, self).__init__()

        filters = int(input_filters * expand_ratio)
        # https://github.com/google/automl/blob/5b2b61005c54efe72f97cf7d0e924afed6a62a60/efficientnetv2/effnetv2_model.py#L241
        reduced_filters = int(input_filters * se_ratio)
        stride = 1

        self.downsample = downsample
        if self.downsample:
            stride = 2
            self.pool = MaxPool2d(kernel_size=stride, stride=stride, pad_mode='same')
            self.proj = nn.Conv2d(input_filters, output_filters, kernel_size=1, has_bias=True)

        self.norm = nn.BatchNorm2d(input_filters, eps=bn_eps, momentum=bn_momentum)

        self.conv = nn.SequentialCell([
            # expand convolution (strided conv)
            nn.Conv2d(input_filters, filters, kernel_size=1, stride=stride, has_bias=False),
            # nn.Conv2d(input_filters, filters, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(filters, eps=bn_eps, momentum=bn_momentum),
            nn.GELU(),
            # depth-wise convolution (non-strided conv)
            nn.Conv2d(filters, filters, kernel_size=3, group=filters, has_bias=False),
            # nn.Conv2d(filters, filters, kernel_size=3, stride=stride, group=filters, has_bias=False),
            nn.BatchNorm2d(filters, eps=bn_eps, momentum=bn_momentum),
            nn.GELU(),
            # squeeze and excitation
            SE(filters, reduced_filters),
            # projection
            nn.Conv2d(filters, output_filters, kernel_size=1, has_bias=True),
        ])
        self.drop_path = DropPath(survival_prob=1.0-drop_path_rate)

    def construct(self, x):
        if self.downsample:
            # x ← Proj(Pool(x)) + Conv(DepthConv(Conv(Norm(x), stride = 2))))
            x = self.proj(self.pool(x)) + self.drop_path(self.conv(self.norm(x)))
        else:
            # x ← x + Module(Norm(x))
            x = x + self.drop_path(self.conv(self.norm(x)))
        return x


class NCHWToNHWC(nn.Cell):
    """NCHWToNHWC"""
    def __init__(self):
        super(NCHWToNHWC, self).__init__()

    def construct(self, x):
        x = ms.ops.Transpose()(x, (0, 2, 3, 1))
        return x


class NHWCToNLC(nn.Cell):
    """NHWCToNLC"""
    def __init__(self):
        super(NHWCToNLC, self).__init__()

    def construct(self, x):
        x_shape = ms.ops.Shape()(x)
        x = ms.ops.Reshape()(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))
        return x


class NLCToNHWC(nn.Cell):
    """NLCToNHWC"""
    def __init__(self, resolution):
        super(NLCToNHWC, self).__init__()
        self.h, self.w = resolution

    def construct(self, x):
        x_shape = ms.ops.Shape()(x)
        x = ms.ops.Reshape()(x, (x_shape[0], self.h, self.w, x_shape[2]))
        x = ms.ops.Transpose()(x, (0, 3, 1, 2))
        return x


class RelativePositionBias(nn.Cell):
    """RelativePositionBias"""
    def __init__(self, resolution, heads):
        super(RelativePositionBias, self).__init__()

        h, w = resolution
        count = (2 * h - 1) * (2 * w - 1)
        coords = np.meshgrid(np.arange(h), np.arange(w))
        coords_flatten = np.stack(coords).transpose(
            (0, 2, 1)).flatten().reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords[0] += h - 1
        relative_coords[1] += w - 1
        relative_coords[0] *= 2 * w - 1
        relative_coords = relative_coords.sum(0).flatten()

        self.h = h
        self.w = w
        self.heads = heads
        self.count = count
        self.on_value = ms.Tensor(1.0, ms.dtype.float32)
        self.off_value = ms.Tensor(0.0, ms.dtype.float32)
        self.relative_position_index = ms.Tensor(relative_coords, dtype=ms.dtype.int32)
        self.relative_position_bias_table = ms.Parameter(
            ms.common.initializer.initializer(
                "TruncatedNormal", shape=(heads, 2 * h - 1, 2 * w - 1), dtype=ms.dtype.float32),
            name='relative_position_bias_table', requires_grad=True)

    def construct(self):
        relative_position_index = ms.ops.OneHot()(
            self.relative_position_index,
            self.count,
            self.on_value,
            self.off_value
        )
        relative_position_bias_table = ms.ops.Reshape()(self.relative_position_bias_table, (self.heads, self.count))
        relative_position_bias = ms.ops.MatMul(False, True)(relative_position_index, relative_position_bias_table)
        relative_position_bias = ms.ops.Reshape()(relative_position_bias, (1, self.h * self.w, self.h * self.w, self.heads))
        relative_position_bias = ms.ops.Transpose()(relative_position_bias, (0, 3, 1, 2))
        return relative_position_bias


class Attention(nn.Cell):
    """Attention"""
    def __init__(self, input_dim, output_dim, resolution,
                 head_dim=32, dropout_rate=0.0):
        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.heads = output_dim // head_dim
        self.h, self.w = resolution
        self.sqrt_scale = ms.Tensor(head_dim ** -0.25, dtype=ms.dtype.float32)

        self.relative_position_bias = RelativePositionBias((self.h, self.w), self.heads)
        self.q = nn.Dense(input_dim, output_dim, has_bias=True)
        self.k = nn.Dense(input_dim, output_dim, has_bias=True)
        self.v = nn.Dense(input_dim, output_dim, has_bias=True)

        self.softmax = nn.Softmax()
        self.proj = nn.Dense(output_dim, output_dim, has_bias=True)
        self.attn_dropout = nn.Dropout(keep_prob=1.0-dropout_rate)
        self.proj_dropout = nn.Dropout(keep_prob=1.0-dropout_rate)

    def construct(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = ms.ops.Cast()(self.sqrt_scale, ms.ops.DType()(q)) * ms.ops.Reshape()(q, (-1, self.h * self.w, self.heads, self.head_dim))
        q = ms.ops.Transpose()(q, (0, 2, 1, 3)) # batch_size, heads, h*w, head_dim
        k = ms.ops.Cast()(self.sqrt_scale, ms.ops.DType()(k)) * ms.ops.Reshape()(k, (-1, self.h * self.w, self.heads, self.head_dim))
        k = ms.ops.Transpose()(k, (0, 2, 3, 1)) # batch_size, heads, head_dim, h*w
        v = ms.ops.Reshape()(v, (-1, self.h * self.w, self.heads, self.head_dim))
        v = ms.ops.Transpose()(v, (0, 2, 1, 3)) # batch_size, heads, h*w, head_dim
        attn = ms.ops.BatchMatMul()(q, k) # batch_size, heads, h*w, h*w
        relative_position_bias = self.relative_position_bias()
        attn = attn + relative_position_bias
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        attn = ms.ops.BatchMatMul()(attn, v) # batch_size, heads, h*w, head_dim
        attn = ms.ops.Transpose()(attn, (0, 2, 1, 3)) # batch_size, h*w, heads, head_dim
        # attn = ms.ops.Reshape()(attn, (-1, self.h * self.w, self.output_dim))
        attn = ms.ops.Reshape()(attn, (-1, self.h, self.w, self.output_dim))
        attn = self.proj(attn)
        attn = self.proj_dropout(attn)
        return attn


class FeedForward(nn.Cell):
    """FeedForward"""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()

        self.ffn = nn.SequentialCell([
            nn.Dense(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(keep_prob=1.0-dropout_rate),
            nn.Dense(hidden_dim, input_dim),
            nn.Dropout(keep_prob=1.0-dropout_rate)
        ])

    def construct(self, x):
        x = self.ffn(x)
        return x


class Transformer(nn.Cell):
    """Transformer"""
    def __init__(self, input_dim, output_dim, resolution, drop_path_rate, downsample=False,
                 head_dim=32, expand_ratio=4, dropout_rate=0.0, ln_epsilon=0.00001):
        super(Transformer, self).__init__()

        self.downsample = downsample
        if self.downsample:
            stride = 2
            self.pool = MaxPool2d(kernel_size=stride, stride=stride, pad_mode='same', data_format="NHWC")
            self.proj = nn.Dense(input_dim, output_dim, has_bias=True)

        self.nhwc_to_nlc = NHWCToNLC()
        self.nlc_to_nhwc = NLCToNHWC(resolution)

        self.attn_norm = nn.LayerNorm((input_dim,), epsilon=ln_epsilon)
        self.attn = Attention(input_dim, output_dim, resolution, head_dim, dropout_rate)

        self.ffn_norm = nn.LayerNorm((output_dim,), epsilon=ln_epsilon)
        self.ffn = FeedForward(output_dim, int(output_dim * expand_ratio), dropout_rate)

        self.attn_drop_path = DropPath(survival_prob=1.0-drop_path_rate)
        self.ffn_drop_path = DropPath(survival_prob=1.0-drop_path_rate)

    def construct(self, x):
        if self.downsample:
            # x ← Proj(Pool(x)) + Attention(Pool(Norm(x)))
            x = self.proj(self.pool(x)) + self.attn_drop_path(self.attn(self.pool(self.attn_norm(x))))
        else:
            # x ← x + Module(Norm(x))
            x = x + self.attn_drop_path(self.attn(self.attn_norm(x)))
        # x ← x + Module(Norm(x))
        x = x + self.ffn_drop_path(self.ffn(self.ffn_norm(x)))

        return x


class CoAtNet(nn.Cell):
    """CoAtNet"""
    def __init__(self, resolution, input_channels, depths, channels, drop_path_rate, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super(CoAtNet, self).__init__()

        h, w = resolution
        block = {'C': MBConv, 'T': Transformer}

        drop_path_rates = np.linspace(0, drop_path_rate, sum(depths))

        self.s0 = Stem(input_channels, channels[0])
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], depths[0], (h // 4, w // 4), drop_path_rates[sum(depths[:0]):sum(depths[:1])])
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], depths[1], (h // 8, w // 8), drop_path_rates[sum(depths[:1]):sum(depths[:2])])
        self.nchw_to_nhwc = NCHWToNHWC()
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], depths[2], (h // 16, w // 16), drop_path_rates[sum(depths[:2]):sum(depths[:3])])
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], depths[3], (h // 32, w // 32), drop_path_rates[sum(depths[:3]):sum(depths[:4])])
        self.avgpool = GlobalAvgPool2d(keep_dims=False, data_format="NHWC")
        self.final_layer_norm = nn.LayerNorm((channels[-1],), epsilon=0.00001)
        self.cls_head = nn.SequentialCell([
            nn.Dense(channels[-1], channels[-1], has_bias=True),
            nn.Tanh(),
            nn.Dense(channels[-1], num_classes, has_bias=True)
        ])

    def _make_layer(self, block, input_filters, output_filters, depth, resolution, drop_path_rates):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(block(input_filters, output_filters, resolution, drop_path_rates[i], downsample=True))
            else:
                layers.append(block(output_filters, output_filters, resolution, drop_path_rates[i]))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.nchw_to_nhwc(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.avgpool(x)
        x = self.final_layer_norm(x)
        x = self.cls_head(x)
        return x


def coatnet_0(image_height=224, image_width=224, input_channels=3, drop_path_rate=0.2, num_classes=1000):
    depths = [2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    return CoAtNet((image_height, image_width), input_channels, depths, channels, drop_path_rate=drop_path_rate, num_classes=num_classes)

def coatnet_1(image_height=224, image_width=224, input_channels=3, drop_path_rate=0.3, num_classes=1000):
    depths = [2, 6, 14, 2]
    channels = [64, 96, 192, 384, 768]
    return CoAtNet((image_height, image_width), input_channels, depths, channels, drop_path_rate=drop_path_rate, num_classes=num_classes)

def coatnet_2(image_height=224, image_width=224, input_channels=3, drop_path_rate=0.5, num_classes=1000):
    depths = [2, 6, 14, 2]
    channels = [128, 128, 256, 512, 1024]
    return CoAtNet((image_height, image_width), input_channels, depths, channels, drop_path_rate=drop_path_rate, num_classes=num_classes)

# ms.set_seed(1)
# x = ms.Tensor(np.random.randn(1,3,224,224), dtype=ms.float32)
# net = coatnet_0()
# print(net(x))

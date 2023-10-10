# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
MindSpore implementation of `ShuffleNetV1`.
Refer to ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

__all__ = [
    "ShuffleNetV1",
    "shufflenet_v1_g3_x0_5",
    "shufflenet_v1_g3_x1_0",
    "shufflenet_v1_g3_x1_5",
    "shufflenet_v1_g3_x2_0",
    "shufflenet_v1_g8_x0_5",
    "shufflenet_v1_g8_x1_0",
    "shufflenet_v1_g8_x1_5",
    "shufflenet_v1_g8_x2_0"
]


def _default_cfgs(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
    }


model_cfgs = {
    'shufflenet_v1_g3_x1_0': _default_cfgs(url='shufflenet_v1_g3_x1_0.ckpt'),
}


# fixme: MindSpore support group convolution. Remove redundant implementations.


class ShuffleV1Block(nn.Cell):
    """Basic block of ShuffleNetV1. 1x1 GC -> CS -> 3x3 DWC -> 1x1 GC"""

    def __init__(self, inp, oup, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup
        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, pad_mode="pad", padding=0,
                      group=1 if first_group else group, has_bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        ]
        # todo: In https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/blocks.py#L28,
        #  `dw`(depth-wise conv) is in `branch_main_1`, so before `channel shuffle`. Bug or Feature?
        branch_main_2 = [
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad_mode='pad', padding=pad,
                      group=mid_channels, has_bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, pad_mode="pad", padding=0, group=group, has_bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.SequentialCell(branch_main_1)
        self.branch_main_2 = nn.SequentialCell(branch_main_2)
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.relu = nn.ReLU()
        self.concat = ops.Concat(1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            out = self.relu(x_proj + x)
        else:
            # todo: In https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/blocks.py#L53,
            #  `return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)` is actually same with the following line.
            #  But it's more efficient, for saving the cost of the activation operation on the activated `x_proj`
            out = self.relu(self.concat((self.branch_proj(x_proj), x)))
        return out

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.shape
        # fixme: [RuntimeError: ParseStatement] Unsupported statement 'Assert'.
        # assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        x = self.reshape(x, (batch_size, group_channels, self.group, height, width))
        x = self.transpose(x, (0, 2, 1, 3, 4))
        x = self.reshape(x, (batch_size, num_channels, height, width))
        return x


class ShuffleNetV1(nn.Cell):
    r"""ShuffleNetV1 model class, based on
    `"ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" <https://arxiv.org/abs/1707.01083>`_

    Args:
        n_class (int) : number of classification classes.
        model_size (str) : scale factor which controls the number of channels.
        group (int) : number of group for group convolution.
    """

    def __init__(self, n_class=1000, model_size='2.0x', group=3):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell(
            nn.Conv2d(3, input_channel, 3, 2, 'pad', 1, has_bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(ShuffleV1Block(input_channel, output_channel,
                                               group=group, first_group=first_group,
                                               mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.SequentialCell(features)
        self.global_pool = nn.AvgPool2d(7)
        # fixme: In https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/network.py#L63,
        #  bias is disabled for linear layer of classifier. However, ModelZoo does not!
        self.classifier = nn.Dense(self.stage_out_channels[-1], n_class, has_bias=False)
        self._initialize_weights()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.first_conv(x)
        x = self.max_pool(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = self.reshape(x, (-1, self.stage_out_channels[-1]))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    m.weight.set_data(init.initializer(init.Normal(0.01, 0), m.weight.shape))
                else:
                    m.weight.set_data(init.initializer(init.Normal(1.0 / m.weight.shape[1], 0), m.weight.shape))
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.gamma.set_data(init.initializer(init.Constant(1), m.gamma.shape))
                if m.beta is not None:
                    m.beta.set_data(init.initializer(init.Constant(0.0001), m.beta.shape))
                m.moving_mean.set_data(init.initializer(init.Constant(0), m.moving_mean.shape))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(init.Normal(0.01, 0), m.weight.shape))
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))


def shufflenet_v1_g3_x0_5(**kwargs):
    return ShuffleNetV1(group=3, model_size='0.5x', **kwargs)


def shufflenet_v1_g3_x1_0(**kwargs):
    return ShuffleNetV1(group=3, model_size='1.0x', **kwargs)


def shufflenet_v1_g3_x1_5(**kwargs):
    return ShuffleNetV1(group=3, model_size='1.5x', **kwargs)


def shufflenet_v1_g3_x2_0(**kwargs):
    return ShuffleNetV1(group=3, model_size='2.0x', **kwargs)


def shufflenet_v1_g8_x0_5(**kwargs):
    return ShuffleNetV1(group=8, model_size='0.5x', **kwargs)


def shufflenet_v1_g8_x1_0(**kwargs):
    return ShuffleNetV1(group=8, model_size='1.0x', **kwargs)


def shufflenet_v1_g8_x1_5(**kwargs):
    return ShuffleNetV1(group=8, model_size='1.5x', **kwargs)


def shufflenet_v1_g8_x2_0(**kwargs):
    return ShuffleNetV1(group=8, model_size='2.0x', **kwargs)


if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor

    model = shufflenet_v1_g3_x1_0()
    print(model)
    dummy_input = Tensor(np.random.rand(8, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)

#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]


class VGGNet():
    def __init__(self, layers=16):
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        decode_ends=[]
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        conv1 = self.conv_block(input, 64, nums[0], name="conv1_")
        decode_ends.append(conv1)
        conv2 = self.conv_block(conv1, 128, nums[1], name="conv2_")
        decode_ends.append(conv2)
        conv3 = self.conv_block(conv2, 256, nums[2], name="conv3_")
        decode_ends.append(conv3)
        conv4 = self.conv_block(conv3, 512, nums[3], name="conv4_")
        decode_ends.append(conv4)
        conv5 = self.conv_block(conv4, 512, nums[4], name="conv5_")
        decode_ends.append(conv5)
        return decode_ends

    def conv_block(self, input, num_filter, groups, name=None):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False)
        return fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16)
    return model


def VGG19():
    model = VGGNet(layers=19)
    return model
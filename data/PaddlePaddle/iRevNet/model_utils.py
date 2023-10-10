"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR 2018
Refer to https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
"""

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg

from paddle.fluid.framework import ParamBase

def split(x):
    n = int(x.shape[1]/2)
    x1 = x[:, :n, :, :]
    x2 = x[:, n:, :, :]
    return x1, x2


def merge(x1, x2):
    return layers.concat([x1, x2], 1)


class injective_pad(dg.Layer):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = lambda x: layers.pad2d(x, [0,pad_size,0,0], pad_value=0.0)

    def forward(self, x):
        x = layers.transpose(x, [0, 2, 1, 3])
        x = self.pad(x)
        return layers.transpose(x, [0, 2, 1, 3])

    def inverse(self, x):
        return x[:, :x.shape[1] - self.pad_size, :, :]


class psi(dg.Layer):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return layers.reshape(layers.transpose(layers.reshape(input, [bs, bl, bl, new_d, h, w]), [0, 3, 4, 1, 5, 2]), [bs, new_d, h * bl, w * bl])

    def forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return layers.reshape(layers.transpose(layers.reshape(input, [bs, d, new_h, bl, new_w, bl]), [0, 3, 5, 1, 2, 4]), [bs, d * bl_sq, new_h, new_w])


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, dg.Layer):
            raise ValueError('Not a Module')
        else:
            self.module.add_sublayer(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, ParamBase):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


def BatchNorm(*args, momentum=0.1, affine=True, param_attr=None, bias_attr=None, **kwargs):
    if not affine:
        param_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=1.0), trainable=False)
        bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0), trainable=False)
    return dg.BatchNorm(*args, momentum=momentum, param_attr=param_attr, bias_attr=bias_attr, **kwargs)


class ReLU(dg.Layer):
    def forward(self, input):
        return layers.relu(input)
        

class CrossEntropyLoss(dg.Layer):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        super().__init__()
        
        self.nll_loss = dg.NLLLoss(weight, reduction, ignore_index)
    def forward(self, input, target):
        return self.nll_loss(layers.log_softmax(input, 1), target)
            
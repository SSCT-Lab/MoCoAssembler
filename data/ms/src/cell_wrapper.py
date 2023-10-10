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
"""Cell_wrapper"""


import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


_ema_op = C.MultitypeFuncGraph("_ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _update_ema(ema_decay, shadow_variables, variables):
    return P.Assign()(shadow_variables,
                      shadow_variables * ema_decay + variables * (1.0 - ema_decay))

class EMACell(nn.Cell):
    def __init__(self, variables, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.shadow_variables = variables.clone(prefix="ema")
        self.ema_decay = ms.Tensor(ema_decay, ms.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, variables):
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay),
                                 self.shadow_variables,
                                 variables)
        return success


_grad_scale = C.MultitypeFuncGraph("_grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(P.Reciprocal()(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return ms.common.RowTensor(grad.indices,
                               grad.values * F.cast(P.Reciprocal()(scale), F.dtype(grad.values)),
                               grad.dense_shape)


class CustomTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense,
                 enable_ema=False, ema_decay=0.9999,
                 enable_clip_norm=False, gradient_norm=1.0):
        super(CustomTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_sense)
        self.enable_ema = enable_ema
        self.ema_decay = ema_decay
        self.enable_clip_norm = enable_clip_norm
        self.gradient_norm = gradient_norm
        self.print = ms.ops.Print()

        if self.enable_ema:
            params = []
            for _, param in self.network.parameters_and_names():
                params.append(param)
            self.params = ms.ParameterTuple(params)
            self.ema_variables = EMACell(self.params, ema_decay=ema_decay)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if overflow:
            self.print("WARNING: Overflow detected, current loss scale is", self.scale_sense)
        # if there is no overflow, do optimize
        if not overflow:
            # clip grad norm
            if self.enable_clip_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.gradient_norm)
            loss = F.depend(loss, self.optimizer(grads))
            # update ema parameters
            if self.enable_ema:
                self.ema_variables(self.params)
        return loss, cond, scaling_sens


class CustomTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens,
                 enable_ema=False, ema_decay=0.9999,
                 enable_clip_norm=False, gradient_norm=1.0):
        super(CustomTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.enable_ema = enable_ema
        self.ema_decay = ema_decay
        self.enable_clip_norm = enable_clip_norm
        self.gradient_norm = gradient_norm

        if self.enable_ema:
            params = []
            for _, param in self.network.parameters_and_names():
                params.append(param)
            self.params = ms.ParameterTuple(params)
            self.ema_variables = EMACell(self.params, ema_decay=ema_decay)
            
    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        if self.enable_clip_norm:
            grads = C.clip_by_global_norm(grads, clip_norm=self.gradient_norm)
        loss = F.depend(loss, self.optimizer(grads))
        if self.enable_ema:
            self.ema_variables(self.params)
        return loss

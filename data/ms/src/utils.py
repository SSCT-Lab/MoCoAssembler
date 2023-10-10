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


import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm


def get_lr(global_step, lr_max, lr_min, lr_warmup_epochs,
           total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): current global step
       lr_max(float): maximum learning rate
       lr_min(float): minimum learning rate
       lr_decay(float): learning rate decay rate
       lr_warmup_epochs(float): number of epochs for learning rate warm-up
       lr_num_epochs_per_decay(float): number of epochs per decay
       total_epochs(int): total training epochs
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    warmup_steps = int(steps_per_epoch * lr_warmup_epochs)
    annealing_steps = int(steps_per_epoch * total_epochs) - warmup_steps
    lr_max = float(lr_max)
    lr_min = float(lr_min)
    for i in range(steps_per_epoch * total_epochs):
        if i < warmup_steps:
            lr = lr_min + (lr_max - lr_min) * i / warmup_steps
        else:
            # lr = lr_init * lr_decay ** ((i - warmup_steps) // decay_steps)
            lr = lr_min + (lr_max - lr_min) * 0.5 * (1.0 + np.cos((i - warmup_steps) / annealing_steps * np.pi))
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """load checkpoint into network."""
    param_dict = ms.load_checkpoint(pretrain_ckpt_path)
    ms.load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False


def context_device_init(config):
    config.rank_id = 0
    config.rank_size = 1

    if config.device_target == "CPU":
        ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

    elif config.device_target in ["Ascend", "GPU"]:
        ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                               enable_graph_kernel=config.enable_graph_kernel, device_id=config.device_id)
        if config.run_distribute:
            comm.init()
            config.rank_id = comm.get_rank()
            config.rank_size = comm.get_group_size()
            ms.context.set_auto_parallel_context(device_num=config.rank_size,
                                                 parallel_mode=ms.context.ParallelMode.DATA_PARALLEL,
                                                 gradients_mean=True)
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return ms.ops.functional.cast(self._op(x), ms.dtype.float16)


def do_keep_cell_fp32(network, cell_types=(nn.Softmax, nn.LayerNorm)):
    """Do keep cell fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, cell_types):
            ms.log.warning('{} is kept fp32'.format(subcell))
            network._cells[name] = OutputTo16(subcell.to_float(ms.dtype.float32))
            change = True
        else:
            do_keep_cell_fp32(subcell, cell_types)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def build_params_groups(net, weight_decay):
    """build params groups"""
    decayed_params = []
    undecayed_params = []
    for param in net.trainable_params():
        # if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
        if param.name.endswith('.weight'):
            decayed_params.append(param)
        else:
            undecayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': undecayed_params, 'weight_decay': 0.0},
                    {'order_params': net.trainable_params()}]
    return group_params


def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params

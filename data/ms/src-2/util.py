# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Util class or function."""
import os
import stat
import copy
import numpy as np

from copy import deepcopy
from src.config import cfg
# from sklearn.metrics import confusion_matrix

import mindspore.nn as nn
import mindspore.ops as P
import mindspore.common.dtype as mstype

from mindspore.train.model import Model
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.callback import Callback
from mindspore import Tensor, Parameter
from mindspore.communication.management import GlobalComm
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore import load_checkpoint


def load_backbone(net, ckpt_path):
    """Load BNInception backbone checkpoint."""
    # param_dict = np.load(ckpt_path, allow_pickle=True).item()
    param_dict = load_checkpoint(ckpt_path)  # 直接返回的就是一个dict

    for k, v in param_dict.items():
        param_dict[k] = Parameter(Tensor(v, mstype.float32), name='w')

    bninception_prefix = 'base_model.'
    find_param = []
    not_found_param = []
    net.init_parameters_data()
    for name, cell in net.cells_and_names():
        if name.startswith(bninception_prefix):
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                darknet_weight = '{}.weight'.format(name)
                darknet_bias = '{}.bias'.format(name)
                if darknet_weight in param_dict:
                    cell.weight.set_data(param_dict[darknet_weight].data)
                    find_param.append(darknet_weight)
                else:
                    not_found_param.append(darknet_weight)
                if darknet_bias in param_dict:
                    cell.bias.set_data(param_dict[darknet_bias].data)
                    find_param.append(darknet_bias)
                else:
                    not_found_param.append(darknet_bias)
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                darknet_moving_mean = '{}.moving_mean'.format(name)
                darknet_moving_variance = '{}.moving_variance'.format(name)
                darknet_gamma = '{}.gamma'.format(name)
                darknet_beta = '{}.beta'.format(name)
                if darknet_moving_mean in param_dict:
                    cell.moving_mean.set_data(param_dict[darknet_moving_mean].data)
                    find_param.append(darknet_moving_mean)
                else:
                    not_found_param.append(darknet_moving_mean)
                if darknet_moving_variance in param_dict:
                    cell.moving_variance.set_data(param_dict[darknet_moving_variance].data)
                    find_param.append(darknet_moving_variance)
                else:
                    not_found_param.append(darknet_moving_variance)
                if darknet_gamma in param_dict:
                    cell.gamma.set_data(param_dict[darknet_gamma].data)
                    find_param.append(darknet_gamma)
                else:
                    not_found_param.append(darknet_gamma)
                if darknet_beta in param_dict:
                    cell.beta.set_data(param_dict[darknet_beta].data)
                    find_param.append(darknet_beta)
                else:
                    not_found_param.append(darknet_beta)

    # print('================found_param {}========='.format(len(find_param)))
    # print(find_param)
    # print('================not_found_param {}========='.format(len(not_found_param)))
    # print(not_found_param)
    print('====== load {} successfully ======'.format(ckpt_path))
    return net


def process_trainable_params(trainable_params):
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    for x in trainable_params:
        if x.name == "base_model.conv1_7x7_s2.conv1_7x7_s2.weight":
            group1.append(x)
        elif x.name == "base_model.conv1_7x7_s2.conv1_7x7_s2.bias":
            group2.append(x)
        elif "bn" in x.name:
            group5.append(x)
        else:
            if "weight" in x.name:
                group3.append(x)
            else:
                group4.append(x)
    return group1, group2, group3, group4, group5


def get_lr(learning_rate, gamma, epochs, steps_per_epoch, lr_steps):
    lr_each_step = []
    base_lr = learning_rate
    for epoch in range(1, epochs + 1):
        decay = gamma ** (sum(epoch >= np.array(lr_steps)))
        base_lr = base_lr * decay
        # if epoch%lr_steps==0:
        #    base_lr = base_lr*gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(base_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        besk_ckpt_name (str): bast checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.bast_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)
            print("epoch: {}, {}: {}".format(cur_epoch, self.metrics_name, res), flush=True)
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                print("update best result: {}".format(res), flush=True)
                if self.save_best_ckpt:
                    if os.path.exists(self.bast_ckpt_path):
                        self.remove_ckpoint_file(self.bast_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.bast_ckpt_path)
                    print("update best checkpoint at: {}".format(self.bast_ckpt_path), flush=True)

    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch), flush=True)


class ClassifyCorrectCell(nn.Cell):
    r"""
    Cell that returns correct count of the prediction in classification network.
    This Cell accepts a network as arguments.
    It returns orrect count of the prediction to calculate the metrics.

    Args:
        network (Cell): The network Cell.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar correct count of the prediction

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> eval_net = nn.ClassifyCorrectCell(net)
    """

    def __init__(self, network):
        super(ClassifyCorrectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = self.allreduce(y_correct)
        return (total_correct,)


class DistAccuracy(nn.Metric):
    r"""
    Calculates the accuracy for classification data in distributed mode.
    The accuracy class creates two local variables, correct number and total number that are used to compute the
    frequency with which predictions matches labels. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides correct number by total number.

    .. math::

        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}

        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for classification (single-label).

    Examples:
        >>> y_correct = Tensor(np.array([20]))
        >>> metric = nn.DistAccuracy(batch_size=3, device_num=8)
        >>> metric.clear()
        >>> metric.update(y_correct)
        >>> accuracy = metric.eval()
    """

    def __init__(self, batch_size, device_num):
        super(DistAccuracy, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type

        Raises:
            ValueError: If the number of the input is not 1.
        """

        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """

        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num


def load_nparray_into_net(net, array_dict):
    """
    Loads dictionary of numpy arrays into network.

    Args:
        net (Cell): Cell network.
        array_dict (dict): dictionary of numpy array format model weights.
    """
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in array_dict:
            new_param = array_dict[param.name]
            param.set_data(Parameter(Tensor(deepcopy(new_param)), name=param.name))
        else:
            param_not_load.append(param.name)
    return param_not_load


class EmaEvalCallBack(Callback):
    """
    Call back that will evaluate the model and save model checkpoint at
    the end of training epoch.

    Args:
        network: tinynet network instance.
        ema_network: step-wise exponential moving average of network.
        eval_dataset: the evaluation daatset.
        decay (float): ema decay.
        save_epoch (int): defines how often to save checkpoint.
        dataset_sink_mode (bool): whether to use data sink mode.
        start_epoch (int): which epoch to start/resume training.
    """

    def __init__(self, network, ema_network, eval_dataset, loss_fn, decay=0.999, \
                 save_epoch=1, dataset_sink_mode=False, start_epoch=0, interval=2, ckpt_path=""):
        self.network = network
        self.ema_network = ema_network
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.decay = decay
        self.save_epoch = save_epoch
        self.shadow = {}
        self.ema_accuracy = {}

        self.best_ema_accuracy = 0
        self.best_accuracy = 0
        self.best_ema_epoch = 0
        self.best_epoch = 0
        self._start_epoch = start_epoch
        self.interval = interval
        self.ckpt_path = ckpt_path
        self.eval_metrics = {'Validation-Loss': Loss(),
                             'Top1-Acc': Top1CategoricalAccuracy(),
                             'Top5-Acc': Top5CategoricalAccuracy()}
        self.dataset_sink_mode = dataset_sink_mode

    def begin(self, run_context):
        """Initialize the EMA parameters """
        for _, param in self.network.parameters_and_names():
            self.shadow[param.name] = deepcopy(param.data.asnumpy())

    def step_end(self, run_context):
        """Update the EMA parameters"""
        for _, param in self.network.parameters_and_names():
            new_average = (1.0 - self.decay) * param.data.asnumpy().copy() + \
                          self.decay * self.shadow[param.name]
            self.shadow[param.name] = new_average

    def epoch_end(self, run_context):
        """evaluate the model and ema-model at the end of each epoch"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch >= self._start_epoch and (cur_epoch - self._start_epoch) % self.interval == 0:
            save_ckpt = (cur_epoch % self.save_epoch == 0)
            load_nparray_into_net(self.ema_network, self.shadow)
            model = Model(self.network, loss_fn=self.loss_fn, metrics=self.eval_metrics)
            model_ema = Model(self.ema_network, loss_fn=self.loss_fn, metrics=self.eval_metrics)
            acc = model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
            ema_acc = model_ema.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
            print("Model Accuracy:", acc)
            print("EMA-Model Accuracy:", ema_acc)

            output = [{"name": k, "data": Tensor(v)}
                      for k, v in self.shadow.items()]
            self.ema_accuracy[cur_epoch] = ema_acc["Top1-Acc"]
            if self.best_ema_accuracy < ema_acc["Top1-Acc"]:
                self.best_ema_accuracy = ema_acc["Top1-Acc"]
                self.best_ema_epoch = cur_epoch
                # save_checkpoint(output, "ema_best.ckpt")
                save_checkpoint(output, os.path.join(self.ckpt_path,  "ema_best.ckpt"))


            if self.best_accuracy < acc["Top1-Acc"]:
                self.best_accuracy = acc["Top1-Acc"]
                self.best_epoch = cur_epoch
                save_checkpoint(cb_params.train_network, os.path.join(self.ckpt_path, "best.ckpt"))

            print("Best Model Accuracy: %s, at epoch %s" %
                  (self.best_accuracy, self.best_epoch))
            print("Best EMA-Model Accuracy: %s, at epoch %s" %
                  (self.best_ema_accuracy, self.best_ema_epoch))

            if save_ckpt:
                # Save the ema_model checkpoints
                # ckpt = "{}-{}.ckpt".format("ema", cur_epoch)
                # save_checkpoint(output, os.path.join(self.ckpt_path, ckpt))
                # save_checkpoint(output, "ema_last.ckpt")
                save_checkpoint(output, os.path.join(self.ckpt_path, "ema_last.ckpt"))

                # Save the model checkpoints
                # save_checkpoint(cb_params.train_network, os.path.join(self.ckpt_path, "last.ckpt"))

    def end(self, run_context):
        print("Top 10 EMA-Model Accuracies: ")
        count = 0
        for epoch in sorted(self.ema_accuracy, key=self.ema_accuracy.get,
                            reverse=True):
            if count == 10:
                break
            print("epoch: %s, Top-1: %s)" % (epoch, self.ema_accuracy[epoch]))
            count += 1

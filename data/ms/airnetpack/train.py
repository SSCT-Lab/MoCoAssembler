import os
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore import context
from mindspore.train.callback import LossMonitor,SummaryCollector
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig,Callback
from mindspore.train.serialization import load_checkpoint,load_param_into_net
from mindspore.nn.metrics import Metric

class SquareLoss(nn.loss.loss._Loss):
    def __init__(self,reduction='mean'):
        super().__init__(reduction)
        self.square = P.Square()
    def construct(self, data, label):
        y = self.square(data-label)
        return self.get_loss(y)

class AbsLoss(nn.loss.loss._Loss):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.abs = P.Abs()
        self.squeeze = P.Squeeze(-1)
    def construct(self, data, label):
        y = self.abs(data-label)
        return self.squeeze(y)

class ForceAbsLoss(nn.loss.loss._Loss):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.norm = nn.Norm(-1)
        self.reduce_mean = P.ReduceMean()
    def construct(self, pred_force, label_force):
        diff = pred_force - label_force
        loss = self.norm(diff)
        return self.reduce_mean(loss,-1)

class WithForceLossCell(nn.Cell):
    def __init__(self, backbone, energy_fn, force_fn, ratio_energy=0.01):
        super().__init__(auto_prefix=False)
        self._backbone = backbone
        self.force_fn = force_fn
        self.energy_fn = energy_fn
        self.ratio_energy = ratio_energy
        self.ratio_force = 1.0 - ratio_energy
        self.grad_op = C.GradOperation()
    def construct(self, positions, forces, energy):
        out = self._backbone(positions)
        fout = -1 * self.grad_op(self._backbone)(positions)
        loss_force = self.force_fn(fout,forces) * self.ratio_force
        loss_energy = self.energy_fn(out,energy) * self.ratio_energy
        return loss_energy + loss_force
    @property
    def backbone_network(self):
        return self._backbone

class WithForceEvalCell(nn.Cell):
    def __init__(self, network, energy_fn=None, force_fn=None, add_cast_fp32=False):
        super().__init__(auto_prefix=False)
        self._network = network
        self._energy_fn = energy_fn
        self._force_fn = force_fn
        self.add_cast_fp32 = add_cast_fp32

        self.grad_op = C.GradOperation()

    def construct(self, positions, forces, energy):
        outputs = self._network(positions)
        foutputs = -1 * self.grad_op(self._network)(positions)
        if self.add_cast_fp32:
            forces = F.mixed_precision_cast(ms.float32, forces)
            energy = F.mixed_precision_cast(ms.float32, energy)
            outputs = F.cast(outputs, ms.float32)

        if self._energy_fn is None:
            eloss = 0
        else:
            eloss = self._energy_fn(outputs, energy)

        if self._force_fn is None:
            floss = 0
        else:
            floss = self._force_fn(foutputs, forces)
        
        return eloss, floss, outputs, energy, foutputs, forces

class Recorder(Callback):
    def __init__(self, model, filename, per_epoch=1, eval_dataset=None):
        super().__init__()
        if not isinstance(per_epoch, int) or per_epoch < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.model = model
        self._per_epoch = per_epoch
        self.eval_dataset = eval_dataset

        self.filename = filename

        self.last_loss = 0
        self.epoch_loss = 0
        self.sum_loss = 0

        self.ds_num = 0
        self.train_num = 0

        self.record = []

        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(self.filename)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        nbatch = len(cb_params.train_dataset_element[0])
        batch_loss = loss * nbatch

        self.ds_num += nbatch
        self.train_num += nbatch

        self.last_loss = loss
        self.epoch_loss += batch_loss
        self.sum_loss += batch_loss

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num

        self.epoch_loss /= self.ds_num
        mov_avg = self.sum_loss / self.train_num

        info = 'Epoch: ' + str(cur_epoch) + \
             ', Last_Loss: ' + str(self.last_loss) + \
             ', Epoch_Loss: ' + str(self.epoch_loss) + \
             ', Avg_loss: ' + str(mov_avg)

        if self.eval_dataset is not None:
            eval_metrics = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            for k,v in eval_metrics.items():
                info += ', '
                info += k
                info += ': '
                info += str(v)

        with open(self.filename, "a") as f:
            f.write(info + os.linesep)

        if cur_epoch % self._per_epoch == 0:
            print(info, flush=True)
        
        self.epoch_loss = 0
        self.ds_num = 0

    # def end(self, run_context):
    #     np.savetxt(self.filename,np.array(self.record))

class MAE(Metric):
    def __init__(self,indexes=[2,3]):
        super().__init__()
        self.clear()
        self._indexes = indexes

    def clear(self):
        self._abs_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        abs_error = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += np.average(abs_error) * y.shape[0]
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num

class MSE(Metric):
    def __init__(self,indexes=[4,5]):
        super().__init__()
        self.clear()
        self._indexes = indexes

    def clear(self):
        self._abs_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        error = y.reshape(y_pred.shape) - y_pred
        error = np.linalg.norm(error,axis=-1)
        self._abs_error_sum += np.average(error) * y.shape[0]
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num


class MLoss(Metric):
    def __init__(self,index=0):
        super().__init__()
        self.clear()
        self._index = index

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):

        loss = self._convert_data(inputs[self._index])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError("Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num

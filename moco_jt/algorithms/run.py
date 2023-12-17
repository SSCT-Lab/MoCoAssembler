import random
import shutil
import traceback
from importlib import import_module
from datetime import datetime

import jittor

import file_paths
import os
# from filter import Filter
from train import Trainer

t = Trainer()
# os.environ['TRAIN_STOP_FLAG'] = '0'


class Filter:
    def __init__(self):
        self.info_lis = []
        self.addkey('nn.py", line 31, in matmul_transpose')  # shape problem
        self.addkey('nn.py", line 962')  # strange problem
        self.addkey('nn.py", line 957, in execute')  # Conv dim problem
        self.addkey('nn.py", line 959, in execute')  # Conv channels problem
        # self.addkey('Please refer to examples(help(jt.ops.reshape))')  # reshape problem
        self.addkey('nn.py", line 1369, in execute')  # ConvTranspose dim problem
        self.addkey('not enough values to unpack')  # Other dim problems
        self.addkey('too many values to unpack')  # Other dim problems
        self.addkey(') Shape not match, x:')  # Norm Layers shape problem
        # self.addkey('need_sync->num >= 0')  # some -1 shape problem
        # self.addkey('Linear(in_features = -1')  # Linear shape problem: in_features
        self.addkey('input channel needs to be divided by upscale_factor')  # PixelShuffle shape Problem
        # self.addkey('ValueError: math domain error')  # some -1 shape problem
        # self.addkey('No module named')  # some questions about module import...
        self.addkey('MaxUnpool2d.execute() missing 1 required positional argument:')  # as it said
        # self.addkey('out_channels must be divisible by groups')  # as it said
        self.addkey("end_dim should be larger than or equal to start_dim for flatten function")
        # self.addkey('File "/root/miniconda3/envs/myconda/lib/python3.10/site-packages/jittor/__init__.py", line 671, in flatten')
        self.addkey("assert C==i\nAssertion")
        self.addkey("in_channels must be divisible by groups")
        self.addkey("assert C==self.in_channels")

        # level 2 filter: checked errors

        # self.addkey('AttributeError: module \'jittor\' has no attribute \'softplus\'')  # Mish
        # self.addkey('nn.py", line 2160, in execute')  # UpSample
        # self.addkey('nn.py", line 2163, in execute')  # Another UpSample
        # self.addkey('maximun')  # AdaptiveAvgPool3d
        # self.addkey(':1:1:1:i1:o1:s0,')  # MaxPool2d Crush
        # self.addkey('Vary shape should only occur in the first dimension')  # MaxPool2d CYPRTI

    def judge(self, string) -> bool:  # 如果含有以上字符串，那将被过滤掉，返回False，如果是有价值信息，返回True
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True

    def addkey(self, string: str) -> None:
        self.info_lis.append(string)


def run_single_model(model_name: str, model_type: str) -> (bool, int):
    flag = True
    error = ''
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        net = module.go()
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        del module
        del net
    except Exception as e:
        flag = False
        error = error + str(traceback.format_exc())
        trainable_params = 0

    if flag:
        # print(model_name + ' succeed')
        return True, trainable_params
    else:
        f = Filter()
        if not f.judge(error):
            # print(model_name + ' has error, but filtered out...')
            return False, 0
        # print(model_name + ' has error, and witten into log...')
        result = model_name + '  error   :          \n'
        now = datetime.now()
        timenow = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
            now.hour).zfill(2) + ':' + str(now.minute).zfill(2)
        result = result + 'Time:  ' + timenow.replace(' ', '-').replace(':', '-') + '\n'
        result = result + 'Info: \n' + error + '\n\n\n'
        with open(os.path.join(file_paths.MAIN_PATH, 'logs', model_type, 'log.txt'), 'a') as f:
            f.write(result)

        source = os.path.join(file_paths.MUTATED_MODEL_PATH, model_type, model_name)
        target = os.path.join(file_paths.SAVED_MODEL_PATH, model_type)
        new_source = source[:-3] + '_' + timenow.replace(' ', '-').replace(':', '-') + '.py'
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False, 0


def train_single_model(model_name: str, model_type: str) -> bool:
    flag = True
    if os.environ['TRAIN_STOP_FLAG'] == '1':
        return True
    error = ''
    net_name = model_type
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        net = module.go()
        if "-1-" in model_name or "-2-" in model_name or "-3-" in model_name:
            return True
        if '-0-1' not in model_name and t.dataloader_dict[net_name] is not None:
            t.train(net, net_name)
        del module
        del net
    except Exception:
        flag = False
        error = error + str(traceback.format_exc())

    if not flag:
        result = model_name + '  error(train error)   :          \n'
        now = datetime.now()
        timenow = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
            now.hour).zfill(2) + ':' + str(now.minute).zfill(2)
        result = result + 'Time:  ' + timenow + '\n'
        result = result + 'Info: \n' + error + '\n\n\n'
        with open(os.path.join(file_paths.MAIN_PATH, 'logs', model_type, 'log.txt'), 'a') as f:
            f.write(result)

        source = os.path.join(file_paths.MUTATED_MODEL_PATH, model_type, model_name)
        target = os.path.join(file_paths.SAVED_MODEL_PATH, model_type)
        new_source = source[:-3] + '_' + timenow.replace(' ', '-').replace(':', '-') + '.py'
        if 'Something wrong... Could you please report this issue?' in error or 'Async error was detected' in error:
            os.environ['TRAIN_STOP_FLAG'] = '1'  # 注：这是train的保护机制，因为jittor有可能导致后面模型全崩
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False
    else:
        return True

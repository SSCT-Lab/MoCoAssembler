import shutil
import traceback
from datetime import datetime
import random
import time
# from filter import Filter
import torch
import file_paths
import mutate
from train import Trainer
import subprocess
import glob
import file_paths
import os
from importlib import import_module
import re

t = Trainer()


class Filter:
    def __init__(self):
        self.info_lis = []
        self.addkey("must be divisible by groups")
        # self.addkey("pad should be at most half of kernel size")
        self.addkey("missing 1 required positional argument")
        # self.addkey("Kernel size can't be greater than actual input size")
        self.addkey("out_channels must be divisible by groups")
        self.addkey("but got Tensor of dimension")
        self.addkey("mat1 and mat2 shapes cannot be multiplied")
        # self.addkey("Bilinear.forward() missing 1 required positional argument")
        # self.addkey("No module named")
        self.addkey("but got input of size")
        # self.addkey("syntax error")
        self.addkey("Output size is too small")
        self.addkey("expected to be in range of")
        self.addkey("missing 2 required positional arguments")
        self.addkey("channels instead")
        self.addkey("float() argument must be a string or a real number, not ")
        self.addkey("flatten() has invalid args: start_dim cannot come after end_dim")
        self.addkey("Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now")
        self.addkey("'tuple' object has no attribute")
        self.addkey("Expected 2D or 3D (batch mode) tensor for input")
        self.addkey("pool2d(): Expected")
        self.addkey("tensor expected for input")
        self.addkey("while checking arguments for")
        self.addkey("expects input with > 2 dims")
        self.addkey("must be tuple of ints, but found")
        self.addkey("Sizes of tensors must match")
        self.addkey("Tensors must have same number of dimensions")
        self.addkey("must be Tensor, not tuple")
        self.addkey("input has inconsistent input_size")
        self.addkey("input.size(-1) must be equal to input_size")
        self.addkey("It is expected dilation equals to 2")
        self.addkey("Input dimension should be at least 3")
        self.addkey("running_mean should contain")
        self.addkey("weight should contain")
        self.addkey("Expected weight to be")
        self.addkey("Padding length must be divisible by 2")
        self.addkey("It is expected stride equals to 2")
        self.addkey("expected 4D input")
        self.addkey("Expected more than 1 spatial element when training")
        self.addkey("Expected size of input")
        self.addkey("The size of tensor a")

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True

    def addkey(self, string: str) -> None:
        self.info_lis.append(string)


def run_single_model(model_name: str, model_type: str) -> (bool, int):  # 5.25新增，单个模型执行不成功就返回False并把error保存起来
    # 6.26新增，执行后不仅返回结果，同时要返回参数数量
    flag = True
    error = ''
    net_name = model_type
    trainable_params = 0
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        module.go()
        # trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # 7.8 param return closed, net closed
        del module
        # del net
    except Exception:
        flag = False
        error = error + str(traceback.format_exc())

    if not flag:
        f = Filter()
        if not f.judge(error):
            # print('但是被过滤掉了')
            return False, 0
        # print(model_name + ' error')
        # error = error[1202:]
        result = model_name + '  error   :          \n'
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
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False, 0
    else:
        return True, trainable_params


def train_single_model(model_name: str, model_type: str) -> bool:
    flag = True
    error = ''
    net_name = model_type
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        net = module.go()
        if "-1-" in model_name or "-2-" in model_name or "-3-" in model_name:
            return True
        if ('BiLSTM' in model_name or 'LSTM' in model_name or 'GRU' in model_name) and '-1' in model_name:
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
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False
    else:
        return True


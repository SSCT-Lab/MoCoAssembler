import shutil
import traceback
from datetime import datetime
import random
import time
from filter import Filter
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


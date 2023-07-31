import random
import shutil
import traceback
from importlib import import_module
from datetime import datetime

import jittor

import file_paths
import os
from filter import Filter
from train import Trainer

t = Trainer()
# os.environ['TRAIN_STOP_FLAG'] = '0'


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

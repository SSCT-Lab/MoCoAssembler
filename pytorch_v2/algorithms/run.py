import shutil
import traceback
from datetime import datetime
import random

import file_paths
import mutate

import subprocess
import glob
import file_paths
import os
from importlib import import_module
import re


def run_single_model(model_name: str, model_type: str) -> bool:  # 5.25新增，单个模型执行不成功就返回False并把error保存起来
    flag = True
    error = ''
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        module.go()
    except Exception:
        flag = False
        error = error + str(traceback.format_exc())

    if not flag:
        print(model_name + ' error')
        if 'No module named' in error:
            print('但是被过滤掉了')
            return False
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
        return False
    else:
        return True




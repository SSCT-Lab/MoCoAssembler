import shutil
from datetime import datetime
import random

import file_paths
import mutate
import assemble
import simple_model_split

import subprocess
import glob
import file_paths
import os
import re


def run_single_model(model_name: str) -> bool:  # 5.25新增，单个模型执行不成功就返回False并把error保存起来
    process = subprocess.Popen(os.path.join(file_paths.MUTATED_MODEL_PATH, model_name), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    # 计图前面那些编译提示信息全是用error输出的，给我气笑了，这里使用长度判断
    error = str(error)

    if len(error) > 1260:
        print(model_name + ' error\n')
        error = error[1202:]
        result = model_name + '  error   :          \n'
        now = datetime.now()
        timenow = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
            now.hour).zfill(2) + ':' + str(now.minute).zfill(2)
        result = result + 'Time:  ' + timenow + '\n'
        result = result + 'Info: \n' + error + '\n\n\n'
        with open(os.path.join(file_paths.MAIN_PATH, 'log.txt'), 'a') as f:
            f.write(result)

        source = os.path.join(file_paths.MUTATED_MODEL_PATH, model_name)
        target = file_paths.SAVED_MODEL_PATH
        new_source = source[:-3] + '_' + timenow.replace(' ', '-').replace(':', '-') + '.py'
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False
    else:
        return True


if __name__ == '__main__':
    run_single_model('testnet-1-1.py')

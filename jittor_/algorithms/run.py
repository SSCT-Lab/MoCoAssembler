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


def sort_by_numbers(file_name):
    match = re.search(r'\d+', file_name)  # 使用正则表达式提取标号部分
    if match:
        return int(match.group())  # 转换为整数进行比较
    else:
        return float('inf')  # 对于没有标号的文件名，将其排在最后


def run():
    # 指定文件夹路径
    folder_path = file_paths.MUTATED_MODEL_PATH

    # # 构建匹配模式 "*.py"，表示匹配所有以 .py 结尾的文件
    # file_pattern = os.path.join(folder_path, '*.py')
    #
    # # 使用 glob.glob 函数获取匹配的文件列表
    # py_files = glob.glob(file_pattern)

    files = os.listdir(folder_path)
    files = sorted(files, key=sort_by_numbers)
    py_files = []
    for file in files:
        py_files.append(os.path.join(folder_path, file))

    # 遍历所有匹配的 .py 文件，使用 subprocess 执行它们
    index = 0
    for file in py_files:
        if index < 25:
            print(file.lower())
            subprocess.run(['python', file])
            index = index + 1
        else:
            break


def run_single_model(model_name: str) -> bool:  # 5.25新增，单个模型执行不成功就返回False并把error保存起来
    process = subprocess.Popen(os.path.join(file_paths.MUTATED_MODEL_PATH, model_name), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    # 计图前面那些编译提示信息全是用error输出的，给我气笑了，这里使用长度判断
    error = str(error)

    if len(error) > 745:
        print(model_name + ' error\n')
        error = error[741:]
        result = model_name + '  error   :          \n'
        now = datetime.now()
        timenow = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
            now.hour).zfill(2) + ':' + str(now.minute).zfill(2)
        result = result + 'Time:  ' + timenow + '\n'
        result = result + 'Info: \n' + error + '\n\n\n'
        with open(os.path.join(file_paths.MAIN_PATH, 'log.txt'), 'a') as f:
            f.write(result)
        return False
    else:
        return True

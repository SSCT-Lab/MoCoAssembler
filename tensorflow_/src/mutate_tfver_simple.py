# -*- coding: utf-8 -*-

"""
@Title   : inception块处理
@Time    : 2023/5/6 16:34
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""

import json
import time
from queue import Queue
from shutil import copyfile

import yaml

from config.keywords import INF
from config.paths import tf_param_file, tf_func_sim_file, tf_func_file, tf_res_file
from pathlib import Path
import random
import re

Inception = {}


def mutate(model_name):
    """
    模型变异主体
    :param model_name: 模型名称
    :return:
    """

    res_model_dir = tf_res_file / model_name
    mutate_dir = res_model_dir / "mutate"
    # 如果变异文件不存在，则新建
    if not Path.exists(mutate_dir):
        Path.mkdir(mutate_dir)

    input_file_name = res_model_dir.resolve().__str__() + "/" + model_name + "_input.py"
    function_file_name = res_model_dir.resolve().__str__() + "/" + model_name + "_function.py"
    output_file_name = res_model_dir.resolve().__str__() + "/" + model_name + "_output.py"
    inception_file_name = res_model_dir.resolve().__str__() + "/" + model_name + "_inception.py"

    function_file = Path.open(Path(function_file_name), "r", encoding="utf8")

    queue = Queue()
    queue.put(input_file_name)

    # 函数突变体
    mutate_on_model(queue, function_file, mutate_dir)

    # 处理自定义快
    global Inception
    if Inception is None: pass
    else:
        inception_file = Path.open(Path(inception_file_name), "r", encoding="utf8")
        output_file = Path.open(Path(output_file_name), "a+", encoding="utf8")
        mutate_on_module(inception_file, output_file)
        inception_file.close()
        output_file.close()

    # 给每一个文件写入输出层，如果模型有自定义块也一并写入
    mutate_list = mutate_dir.rglob("*.py")
    for _ in mutate_list:
        mutate_file = Path.open(Path(_), "a+", encoding="utf8")
        output_file = Path.open(Path(output_file_name), "r", encoding="utf8")
        for line in output_file:
            mutate_file.write(line)
        mutate_file.close()
        output_file.close()

    function_file.close()


def mutate_on_model(queue, function_file, mutate_dir):
    """
    对模型开始变异
    :param queue: 可变异文件列表
    :return:
    """

    iteration = 0
    mutate = 2
    num = 0
    tmp_queue = Queue()

    mutate_list = [mutate_on_parma, mutate_on_function]

    with Path.open(tf_func_file / "def.json", "r") as file:
        data = json.load(file)
    api_list = list(data.keys())

    for line in function_file:
        if line == "\n" or line.find("#") >= 0:
            pass
        else:
            iteration = iteration + 1

            function = get_function(line)
            if function == line:
                pass
            elif "tf." + function not in api_list:
                global Inception
                if function in Inception.keys():
                    count = Inception[function]
                    Inception[function] = count + 1
                else:
                    Inception[function] = 1

            func_file = "tf." + function + ".yaml"
            while not queue.empty():
                input_file_name = queue.get()

                for i in range(1, mutate + 1):
                    num = num + 1
                    new_mutate_name = mutate_dir.resolve().__str__() + "/" + iteration.__str__() + "_" + num.__str__() + ".py"
                    tmp_queue.put(new_mutate_name)
                    print(new_mutate_name)
                    copyfile(input_file_name, new_mutate_name)
                    file_mut = Path.open(Path(new_mutate_name), "a+")
                    if "tf." + function in api_list:
                        # 随机选择一种变异方法进行变异
                        method = random.choice(mutate_list)
                        if method == mutate_list[0]:
                            file_mut.write("# 参数列表变异\n")
                            file_mut.write(method(line, func_file))
                        else:
                            file_mut.write("# 函数变异\n")
                            file_mut.write(method(line, func_file))
                    else:
                        # 如果function是自定义块的话，就对对自定块进行变异
                        new_function = function + "_" + Inception[function].__str__()
                        file_mut.write(line.replace(function, new_function))
                        file_mut.close()

            while not tmp_queue.empty():
                queue.put(tmp_queue.get())

        num = 0


def get_function(line: str) -> str:
    """
    根据api找函数调用，如果该行api存在函数调用，则返回函数名（可能是tf库中的，也可能不是）
                            不存在函数调用，返回该行api
    :param line: api
    :return:
    """
    try:
        function = re.findall(r".*? = (.*?)\(.*?", line)[0]
    except:
        return line
    return function


def get_params(line: str) -> dict:
    """
    获取参数列表
    :param line: api
    :return: 参数列表
    """
    infos1 = re.findall(r".*?\((?P<param>.*?)\)\((?P<input>.*?)\)", line)

    params = infos1[0][0] + ', '

    infos2 = re.findall(r".*?\((.*?)\).*?", params, re.S)
    for _ in infos2:
        __ = _.replace(" ", "")
        params = params.replace(_, __)

    params_dic = {}
    details = re.finditer(r"(?P<param>.*?)=(?P<value>.*?), ", params, re.S)
    for _ in details:
        params_dic.update({_["param"].strip(): _["value"].strip()})

    return params_dic


def generate_param_line(line, params_dict) -> str:
    """
    根据参数列表生成新的api
    :param line: 一行api
    :param params_dict: 参数列表
    :return: 新生成的api
    """
    new_params: str = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    org_params: str = re.findall(r".*?(\(.*?\))\(.*?", line, re.S)[0]

    new_line = line.replace(org_params, new_params, 1)
    return new_line


def mutate_on_parma(line: str, func_file) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    dict = get_params(line)

    func_file = tf_param_file / func_file
    if func_file.exists():
        with Path.open(func_file, "r") as file:
            data = yaml.load(file, yaml.Loader)
            data = data["constraints"]
            params_dict = dict.copy()
            # 随机选择一个参数进行变异
            param = random.choice(list(data.keys()))
            value = ""
            if "dtype" in data[param]:
                dtype = data[param]["dtype"]
                type = random.choice(dtype) if isinstance(dtype, list) else dtype

                # 参数类型匹配
                match type:
                    case "tf.string":
                        if "enum" in data[param]:
                            value = random.choice((data[param]["enum"]))
                            value = '"' + value + '"'
                        else:
                            value = data[param]["default"]
                            if isinstance(value, str):
                                value = '"' + value + '"'
                    case "tf.bool":
                        value = random.choice([True, False])
                    case "float":
                        value = random.random().__str__()
                    case "int":
                        if "range" in data[param]:
                            # 可能需要改范围
                            value = random.randint(0, INF).__str__()
                        elif "structure" in data[param]:
                            structure = random.choice(data[param]["structure"])
                            # 此处的value值随便取的，应该做一定的更改
                            match structure:
                                case "integer":
                                    value = random.randint(1, 4)
                                case "tuple":
                                    value = tuple(random.randint(1, 4) for _ in range(data[param]["shape"]))
                                case "list":
                                    value = list(random.randint(1, 4) for _ in range(data[param]["shape"]))
                                case "tuple_of_tuples":
                                    value = tuple(tuple(random.randint(1, 4) for _ in range(2)) for _ in
                                                  range(data[param]["shape"]))
                        else:
                            value = data[param]["default"]
            else:
                value = data[param]["default"]
            params_dict[param] = value
            new_line = generate_param_line(line, params_dict)
    else:
        new_line = line
    return new_line


def mutate_on_function(line: str, func_file: str) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    function = get_function(line)
    func_file = tf_func_sim_file / func_file

    # 相似度阈值
    th = 0.6
    if func_file.exists():
        dict = get_params(line)
        with Path.open(func_file, "r") as file:
            data = yaml.load(file, yaml.Loader)
            # 有的函数大于阈值的相似度可能只有它本身，但是变异的时候又不想要他本身，所以简单的处理一下数据
            lst = [_[0] for _ in data.items() if _[1] > th]
            func_mut = random.choice(lst[1:]) if len(lst) > 1 else lst[0]
            param_file = tf_param_file / (func_mut + ".yaml")
            tmp_dict = {}
            if param_file.exists():
                with Path.open(tf_param_file / (func_mut + ".yaml"), "r") as param_file:
                    params_dict = yaml.load(param_file, yaml.Loader)["constraints"]
                    for _ in dict.items():
                        if _[0] in params_dict.keys():
                            tmp_dict[_[0]] = _[1]
            else:
                # 有的函数的参数列表取值可能没有储存，所以直接copy，不改变其参数列表（可能会出错）
                tmp_dict = dict.copy()

            line = line.replace(function, func_mut)
            new_line = generate_param_line(line, tmp_dict)

    return new_line


def mutate_on_module(file_tmp, file_mut):
    mutate_list = [mutate_on_parma, mutate_on_function]
    global Inception
    def_list = []
    for _ in Inception.items():
        function = _[0]
        for line in file_tmp:
            if line.startswith("def " + function):
                def_list.append(line)
                break
        for line in file_tmp:
            def_list.append(line)
            if line.find("return") >= 0:
                break
        # 循环每个自定义块
        for i in range(1, _[1] + 1):
            for line in def_list:
                if line == "\n" or line.find("#") >= 0 or line.find("outputs") >= 0:
                    file_mut.write(line)
                elif line.startswith("def"):
                    new_function = function + "_" + i.__str__()
                    file_mut.write(line.replace(function, new_function))
                else:
                    func = get_function(line)
                    if func == line:
                        file_mut.write(line)
                    else:
                        func_file = "tf." + func + ".yaml"
                        with Path.open(tf_func_file / "def.json", "r") as file:
                            data = json.load(file)
                        api_list = list(data.keys())

                        if "tf." + func in api_list:
                            # 随机选择一种变异方法进行变异
                            method = random.choice(mutate_list)
                            if method == mutate_list[0]:
                                file_mut.write("# 参数列表变异\n")
                                file_mut.write(method(line, func_file))
                            else:
                                file_mut.write("# 函数变异\n")
                                file_mut.write(method(line, func_file))
            file_mut.write("\n\n")


if __name__ == "__main__":
    mutate("a_test")

import json
import re
import subprocess
import traceback
from pathlib import Path
from queue import Queue
import random
from subprocess import call

import yaml
from config.keywords import INPUT_TENSOR, OUTPUT_TENSOR, INF
from config.paths import TF_RES_PATH, TF_FUNC_PATH, TF_PARAM_PATH, TF_FUNC_SIM_PATH, TF_LOG_PATH, TF_MODEL_PATH

import tensorflow as tf
from tensorflow import keras

from tensorflow_.utils.MoCo import MoCo


class MoCoTF(MoCo):
    Inception = {}

    rare_params = ["activity_regularizer",
                   "bias_constraint",
                   "bias_initializer",
                   "bias_regularizer",
                   "data_format",
                   "kernel_constraint",
                   "kernel_initializer",
                   "kernel_regularizer",
                   "depthwise_constraint",
                   "depthwise_initializer",
                   "depthwise_regularizer",
                   "pointwise_constraint",
                   "pointwise_initializer",
                   "pointwise_regularizer",
                   "recurrent_activation",
                   "recurrent_constraint",
                   "recurrent_initializer",
                   "recurrent_regularizer",
                   "beta_constraint",
                   "beta_initializer",
                   "beta_regularizer",
                   "gamma_constraint",
                   "gamma_initializer",
                   "gamma_regularizer",
                   ]

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.res_model_dir = TF_RES_PATH / model_name
        self.mutate_dir = self.res_model_dir / "mutate"
        self.log_file = TF_LOG_PATH

        self.iteration = 0
        self.mutate_times = 2
        self.input_tensor = INPUT_TENSOR
        self.output_tensor = OUTPUT_TENSOR
        self.inf = INF

        self.template_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_template.py"
        self.function_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_function.py"
        self.inception_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_inception.py"

        if not Path.exists(TF_RES_PATH):
            Path.mkdir(TF_RES_PATH)

        if not Path.exists(self.res_model_dir):
            Path.mkdir(self.res_model_dir)

        if not Path.exists(self.mutate_dir):
            Path.mkdir(self.mutate_dir)

        if not Path.exists(self.log_file):
            Path.mkdir(self.log_file)

        with Path.open(TF_FUNC_PATH / "def.json", "r") as file:
            data = json.load(file)
        self.api_list = [_[3:] for _ in data.keys()]

        self.queue = Queue()
        self.queue.put(self.template_file_name)

        self.mutate_list = [self.mutate_on_parma, self.mutate_on_function]
        self.error_list = []

    def depart(self):
        template_file = Path.open(Path(self.template_file_name), "a+", encoding="utf8")
        function_file = Path.open(Path(self.function_file_name), "w", encoding="utf8")
        inception_file = Path.open(Path(self.inception_file_name), "w", encoding="utf8")

        with Path.open(TF_MODEL_PATH / (self.model_name + ".py"), "r", encoding="utf8") as file_org:
            for line in file_org:
                if line.find(self.input_tensor) >= 0:
                    template_file.write("# " + self.model_name + " input layer" + "\n")
                    template_file.write(line)
                    template_file.write("# " + self.model_name + " hidden layer" + "\n")
                    break
                template_file.write(line)

            function_file.write("# " + self.model_name + " function layer" + "\n")
            for line in file_org:
                if line.find(self.output_tensor) >= 0:
                    template_file.write("# " + self.model_name + " output layer" + "\n")
                    template_file.write(line)
                    break
                function_file.write(line)
            function_file.close()

            for line in file_org:
                if line.find("def") >= 0:
                    inception_file.write("# " + self.model_name + " inception layer" + "\n")
                    inception_file.write(line)
                    break
                template_file.write(line)

            for line in file_org:
                if line.find("__name__") >= 0:
                    template_file.write(line)
                    break
                inception_file.write(line)
            inception_file.close()

            for line in file_org:
                template_file.write(line)
            template_file.close()

    def generate_model(self):
        Inception = {}
        new_line = []
        num = 0
        tmp_queue = Queue()

        function_file = Path.open(Path(self.function_file_name), "r", encoding="utf8")

        for line in function_file:
            if line.strip().startswith("#") or line == "\n":
                continue
            else:
                self.iteration += 1

                function = self.get_function(line)
                # print(function)
                if function not in self.api_list:
                    if function in Inception.keys():
                        count = Inception[function]
                        Inception[function] = count + 1
                    else:
                        Inception[function] = 1
                    new_function = function + "_" + Inception[function].__str__()
                    new_line.append(line.replace(function, new_function))
                else:
                    func_file = "tf." + function + ".yaml"
                    for i in range(pow(self.mutate_times, self.iteration)):
                        method = random.choice(self.mutate_list)
                        new_line.append(method(line, func_file))

            while not self.queue.empty():
                org_file_name = self.queue.get()
                with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                    content = org_file.read()
                pos1 = content.find("# " + self.model_name + " output layer")

                for i in range(1, self.mutate_times + 1):
                    num += 1
                    new_file_name = self.mutate_dir.resolve().__str__() + "/" + self.iteration.__str__() + "_" + num.__str__() + ".py"
                    tmp_queue.put(new_file_name)

                    if function not in self.api_list:
                        pos2 = content.find("if __name__")
                        if pos1 != -1 and pos2 != -1:
                            new_module = self.mutate_on_module(function, Inception[function])
                            new_content = content[:pos1] + new_line[0] + content[pos1: pos2] + new_module + content[pos2:]
                            new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                            new_file.write(new_content)
                            new_file.close()

                    else:
                        if pos1 != -1:
                            new_content = content[:pos1] + new_line[num - 1] + content[pos1:]
                            new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                            new_file.write(new_content)
                            new_file.close()

            while not tmp_queue.empty():
                model = tmp_queue.get()

                state = subprocess.call(["python", model])

                if state == 0:
                    self.queue.put(model)
                    print(Path(model).name + "\033[95m运行成功\033[0m")
                else:
                    print(Path(model).name + "\033[94m运行失败\033[0m")
                    self.error_list.append(model)

            num = 0
            new_line = []
            if self.queue.empty():
                break

    def get_function(self, line: str) -> str:
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

    def get_params(self, line: str) -> dict:
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

    def generate_param_line(self, line, params_dict) -> str:
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

    def random_param(self, data) -> str:
        rare_probability = 0.005
        rare_count = 0
        params_list = list(data.keys())
        params_probability = [0 for i in range(len(params_list))]

        for i in range(len(params_list)):
            if params_list[i] in self.rare_params:
                params_probability[i] = rare_probability
                rare_count += 1

        probability = (1 - rare_count * rare_probability) / (len(params_list) - rare_count)

        for i in range(len(params_list)):
            if params_list[i] not in self.rare_params:
                params_probability[i] = probability

        x = random.random()
        cumulative_probability = 0.0
        param = None
        for param, param_probability in zip(params_list, params_probability):
            cumulative_probability += param_probability
            if x < cumulative_probability:
                break

        return param

    def mutate_on_parma(self, line: str, func_file) -> str:
        """
        :param line: 一行api
        :return: 新生成的api
        """
        dict = self.get_params(line)

        func_file = TF_PARAM_PATH / func_file
        if func_file.exists():
            with Path.open(func_file, "r") as file:
                all_data = yaml.load(file, yaml.Loader)
                data = all_data["constraints"]
                params_dict = dict.copy()
                # 随机选择一个参数进行变异

                param = self.random_param(data) if len(list(data.keys())) > 1 else list(data.keys())[0]
                value = ""
                if "dtype" in data[param]:
                    dtype = data[param]["dtype"]
                    type = random.choice(dtype) if isinstance(dtype, list) else dtype
                    if type == "tf.string":
                        if "enum" in data[param]:
                            value = random.choice((data[param]["enum"]))
                            value = '"' + value + '"'
                        else:
                            value = data[param]["default"]
                            if value == "None":
                                pass
                            else:
                                value = '"' + value + '"'
                    elif type == "tf.bool":
                        value = random.choice([True, False])
                    elif type == "float":
                        value = random.random().__str__()
                    elif type == "int":
                        if "structure" in data[param] and "range" in data[param]:
                            structure = data[param]["structure"]
                            structure = random.choice(data[param]["structure"]) if isinstance(structure,
                                                                                              list) else structure
                            drange = data[param]["range"]
                            min_v = int(drange[0])
                            max_v = int(drange[1])

                            if structure == "integer":
                                value = random.randint(min_v, max_v)
                            elif structure == "tuple":
                                value = tuple(random.randint(min_v, max_v) for _ in range(data[param]["shape"]))
                            elif structure == "list":
                                value = list(random.randint(min_v, max_v) for _ in range(data[param]["shape"]))
                            elif structure == "tuple_of_tuples":
                                value = tuple(tuple(random.randint(min_v, max_v) for _ in range(2)) for _ in
                                                  range(data[param]["shape"]))
                        else:
                            value = data[param]["default"]
                else:
                    value = data[param]["default"]
                params_dict[param] = value
                new_line = self.generate_param_line(line, params_dict)
        else:
            new_line = line
        return new_line

    def mutate_on_function(self, line: str, func_file: str) -> str:
        function = self.get_function(line)
        func_file = TF_FUNC_SIM_PATH / func_file

        # 相似度阈值
        th = 0.6
        if func_file.exists():
            dict = self.get_params(line)
            with Path.open(func_file, "r") as file:
                data = yaml.load(file, yaml.Loader)
                # 有的函数大于阈值的相似度可能只有它本身，但是变异的时候又不想要他本身，所以简单的处理一下数据
                lst = [_[0] for _ in data.items() if _[1] > th]
                func_mut = random.choice(lst[1:]) if len(lst) > 1 else lst[0]
                param_file = TF_PARAM_PATH / (func_mut + ".yaml")
                tmp_dict = {}
                if param_file.exists():
                    with Path.open(TF_PARAM_PATH / (func_mut + ".yaml"), "r") as param_file:
                        data = yaml.load(param_file, yaml.Loader)
                        params_dict = data["constraints"]
                        required_list = data["required"]
                        for _ in dict.items():
                            if _[0] in params_dict.keys():
                                tmp_dict[_[0]] = _[1]

                        for _ in required_list:
                            if _ not in tmp_dict:
                                if "default" in params_dict[_]:
                                    tmp_dict[_] = params_dict[_]["default"]
                                else:
                                    tmp_dict[_] = random.randint(params_dict[_]["range"][0], params_dict[_]["range"][1])
                else:
                    # 有的函数的参数列表取值可能没有储存，所以直接copy，不改变其参数列表（可能会出错）
                    tmp_dict = dict.copy()

                line = line.replace(function, func_mut[3:])
                new_line = self.generate_param_line(line, tmp_dict)

        return new_line

    def mutate_on_module(self, function: str, number: int) -> str:
        def_list = []
        inception_file = Path.open(Path(self.inception_file_name), "r", encoding="utf8")

        for line in inception_file:
            if line.startswith("def " + function):
                new_function = function + "_" + number.__str__()
                def_list.append(line.replace(function, new_function))
                break

        for line in inception_file:
            if line.find("return") >= 0:
                def_list.append(line)
                break
            else:
                def_list.append(line)

        # for line in inception_file:
        #     if line.startswith("def " + function):
        #         new_function = function + "_" + number.__str__()
        #         def_list.append(line.replace(function, new_function))
        #     elif line.find("#") >= 0 or line == "\n" or line.find("outputs") >= 0:
        #         def_list.append(line)
        #
        #     else:
        #         func = self.get_function(line)
        #         if func in self.api_list:
        #             func_file = "tf." + func + ".yaml"
        #             method = random.choice(self.mutate_list)
        #             def_list.append(method(line, func_file))
        #         else: def_list.append(line)

        new_module = "".join(def_list)
        return new_module


if __name__ == "__main__":
    test = MoCoTF("lenet")
    # test.depart()
    test.generate_model()

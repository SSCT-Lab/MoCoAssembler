import traceback
from importlib import import_module
from pathlib import Path
import sys

from tqdm import trange, tqdm

sys.path.append(Path.cwd().parent.parent.__str__())

import argparse
import random
import re
import tensorflow as tf
import time
import yaml
from queue import Queue

from config.keywords import INPUT_TENSOR, OUTPUT_TENSOR
from config.paths import RES_PATH, PARAM_PATH, FUNC_SIM_PATH, LOG_PATH, TF_MODEL_PATH, PATH
from utils.MoCo import MoCo



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

    def __init__(self, model_name, mutate_times):
        super().__init__(model_name, mutate_times)
        self.model_name = model_name
        self.res_model_dir = RES_PATH / model_name
        self.log_model_dir = LOG_PATH / model_name
        self.time = int(time.time()).__str__()
        self.ERROR_NUM = 0

        self.mutate_dir = self.res_model_dir / ("mutate" + self.time)
        self.log_dir = self.log_model_dir / ("log" + self.time)

        self.ITERATION = 0
        self.MUTATE_TIMES = mutate_times
        self.NODE_ALL = 1
        self.NODE_ALIVE = 1

        self.template_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_template.py"
        self.function_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_function.py"
        self.inception_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_inception.py"

        if not Path.exists(RES_PATH):
            Path.mkdir(RES_PATH)

        if not Path.exists(self.res_model_dir):
            Path.mkdir(self.res_model_dir)

        if not Path.exists(self.mutate_dir):
            Path.mkdir(self.mutate_dir)

        if not Path.exists(LOG_PATH):
            Path.mkdir(LOG_PATH)

        if not Path.exists(self.log_model_dir):
            Path.mkdir(self.log_model_dir)

        if not Path.exists(self.log_dir):
            Path.mkdir(self.log_dir)

        with Path.open(PATH / "data/api_list.txt", "r") as file:
            self.api_list = [_[3:-1] for _ in file]

        self.queue = Queue()
        self.queue.put(self.template_file_name)

        self.mutate_list = [self.mutate_on_param, self.mutate_on_function]
        self.error_list = []

    def depart(self):
        template_file = Path.open(Path(self.template_file_name), "a+", encoding="utf8")
        function_file = Path.open(Path(self.function_file_name), "w", encoding="utf8")
        inception_file = Path.open(Path(self.inception_file_name), "w", encoding="utf8")

        with Path.open(TF_MODEL_PATH / (self.model_name + ".py"), "r", encoding="utf8") as file_org:
            for line in file_org:
                if line.find(INPUT_TENSOR) >= 0:
                    template_file.write("# " + self.model_name + " input layer" + "\n")
                    template_file.write(line)
                    template_file.write("# " + self.model_name + " hidden layer" + "\n")
                    break
                template_file.write(line)

            function_file.write("# " + self.model_name + " function layer" + "\n")
            for line in file_org:
                if line.find(OUTPUT_TENSOR) >= 0:
                    template_file.write("# " + self.model_name + " output layer" + "\n")
                    template_file.write(line)
                    break
                function_file.write(line)
            function_file.close()

            for line in file_org:
                if line.find("def ") >= 0 and line.find("def go()") < 0:
                    inception_file.write("# " + self.model_name + " inception layer" + "\n")
                    inception_file.write(line)
                    break
                template_file.write(line)

            for line in file_org:
                if line.find("def go()") >= 0:
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
            elif self.get_function(line) == line or line.find("tf.concat") >= 0:
                while not self.queue.empty():
                    org_file_name = self.queue.get()
                    with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                        content = org_file.read()
                        output_pos = content.find("# " + self.model_name + " output layer")
                        if output_pos != -1:
                            new_content = content[:output_pos] + line + content[output_pos:]
                    with Path.open(Path(org_file_name), "w", encoding="utf8") as file:
                        file.write(new_content)
                    tmp_queue.put(org_file_name)

                while not tmp_queue.empty():
                    self.queue.put(tmp_queue.get())
                continue
            else:
                self.ITERATION += 1

                function = self.get_function(line)
                if function not in self.api_list:
                    if function in Inception.keys():
                        count = Inception[function]
                        Inception[function] = count + 1
                    else:
                        Inception[function] = 1
                    new_function = function + "_" + Inception[function].__str__()
                    new_line.append(line.replace(function, new_function))
                else:
                    self.NODE_ALL = self.NODE_ALIVE * self.MUTATE_TIMES

                    for i in range(self.NODE_ALL):
                        method = random.choice(self.mutate_list)
                        new_line.append(method(line))

            while not self.queue.empty():
                org_file_name = self.queue.get()
                with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                    content = org_file.read()
                output_pos = content.find("# " + self.model_name + " output layer")

                for i in range(1, self.MUTATE_TIMES + 1):
                    num += 1
                    new_file_name = self.mutate_dir.resolve().__str__() + "/" + self.ITERATION.__str__() + "_" + num.__str__() + ".py"
                    tmp_queue.put(new_file_name)

                    if function not in self.api_list:
                        __name__pos = content.find("if __name__")
                        if output_pos != -1 and __name__pos != -1:
                            new_module = self.mutate_on_module(function, Inception[function])
                            new_content = content[:output_pos] + new_line[0] + content[output_pos: __name__pos] + new_module + content[
                                                                                                            __name__pos:]
                            new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                            new_file.write(new_content)
                            new_file.close()

                    else:
                        if output_pos != -1:
                            new_content = content[:output_pos] + new_line[num - 1] + content[output_pos:]
                            new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                            new_file.write(new_content)
                            new_file.close()

            self.NODE_ALIVE = 0
            lst = []
            while not tmp_queue.empty():
                lst.append(tmp_queue.get())

            pbar = tqdm(lst, desc="ITERATION: {}".format(self.ITERATION))
            for file_name in pbar:
                pbar.update(self.MUTATE_TIMES)

                try:
                    module_name = '.'.join(file_name.replace("/", ".").split(".")[-6:-1])
                    module = import_module(module_name)
                    module.go()
                    self.queue.put(file_name)
                    self.NODE_ALIVE += 1
                    # print(Path(_).name + "  \033[34mSUCCESS\033[0m")
                except Exception:
                    # print(Path(file_name).name + "  \033[31mFAIL\033[0m")
                    self.ERROR_NUM += 1
                    with Path.open(self.log_dir / Path("error" + self.ERROR_NUM.__str__()), "w", encoding="utf8") as file:
                        file.write(traceback.format_exc())
            del pbar

            num = 0
            new_line = []
            if self.queue.empty():
                break

    def get_function(self, line: str) -> str:
        try:
            function = re.findall(r".*? = (.*?)\(.*?", line)[0]
        except:
            return line
        return function

    def get_params(self, line: str) -> dict:
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

    def generate_line(self, line, params_dict) -> str:
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

    def mutate_on_param(self, line: str) -> str:
        dict = self.get_params(line)
        function = self.get_function(line)
        func_file = PARAM_PATH / ("tf." + function + ".yaml")
        if func_file.exists():
            with Path.open(func_file, "r") as file:
                all_data = yaml.load(file, yaml.Loader)
                data = all_data["constraints"]
                params_dict = dict.copy()

                param = self.random_param(data) if len(list(data.keys())) > 1 else list(data.keys())[0]
                value = self.get_value(data[param])
                params_dict[param] = value
                new_line = self.generate_line(line, params_dict)
        else:
            new_line = line
        return new_line

    def mutate_on_function(self, line: str) -> str:
        function = self.get_function(line)
        func_file = FUNC_SIM_PATH / ("tf." + function + ".yaml")

        th = 0.4
        if func_file.exists():
            dict = self.get_params(line)
            with Path.open(func_file, "r") as file:
                data = yaml.load(file, yaml.Loader)
                lst = [_[0] for _ in data.items() if _[1] > th]
                func_mut = random.choice(lst[1:]) if len(lst) > 1 else lst[0]
                param_file = PARAM_PATH / (func_mut + ".yaml")
                tmp_dict = {}
                if param_file.exists():
                    with Path.open(PARAM_PATH / (func_mut + ".yaml"), "r") as param_file:
                        all_data = yaml.load(param_file, yaml.Loader)
                        data = all_data["constraints"]
                        required_list = all_data["required"]
                        for _ in dict.items():
                            if _[0] in data.keys():
                                tmp_dict[_[0]] = _[1]

                        for _ in required_list:
                            if _ not in tmp_dict:
                                tmp_dict[_] = self.get_value(data[_])
                else:
                    tmp_dict = dict.copy()

                line = line.replace(function, func_mut[3:])
                new_line = self.generate_line(line, tmp_dict)

        return new_line

    def get_value(self, dic):
        value = ""
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "tf.string":
                if "enum" in dic:
                    value = random.choice((dic["enum"]))
                    value = '"' + value + '"'
                else:
                    value = dic["default"]
                    if value == "None":
                        pass
                    else:
                        value = '"' + value + '"'
            elif type == "tf.bool":
                value = random.choice([True, False])
            elif type == "float":
                value = random.random().__str__()
            elif type == "int":
                if "structure" in dic and "range" in dic:
                    structure = dic["structure"]
                    structure = random.choice(dic["structure"]) \
                        if isinstance(structure, list) else structure
                    drange = dic["range"]
                    min_v = int(drange[0])
                    min_v = (min_v + 1) if min_v == 0 else min_v
                    max_v = int(drange[1])

                    if structure == "integer":
                        value = random.randint(min_v, max_v)
                    elif structure == "tuple":
                        value = tuple(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                    elif structure == "list":
                        value = list(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                    elif structure == "tuple_of_tuples":
                        value = tuple(tuple(random.randint(min_v, max_v) for _ in range(2)) for _ in
                                      range(dic["shape"]))
                else:
                    value = dic["default"]
        else:
            value = dic["default"]

        return value

    def mutate_on_module(self, function: str, number: int) -> str:
        def_list = []
        inception_file = Path.open(Path(self.inception_file_name), "r", encoding="utf8")

        func_mut = None
        count = 0

        for line in inception_file:
            if line.startswith("def " + function):
                new_function = function + "_" + number.__str__()
                def_list.append(line.replace(function, new_function))
                break

        for line in inception_file:
            if line.find("#") >= 0 or line == "\n" or line.find("outputs") >= 0:
                def_list.append(line)
            elif line.find("return") >= 0:
                def_list.append(line)
                break
            else:
                func = self.get_function(line)
                if func in self.api_list and func_mut != func and count <= 2:
                    func_mut = func
                    method = random.choice(self.mutate_list)
                    def_list.append(method(line))
                    count += 1
                else:
                    def_list.append(line)

        new_module = "".join(def_list)
        return new_module


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='argparse testing')
    # parser.add_argument('--model_name', '-n', type=str, default="bk", required=True, help="model name")
    # parser.add_argument('--mutate_times', '-t', type=int, default="bk", required=True, help="mutate_times")
    # args = parser.parse_args()
    #
    # test = MoCoTF(args.model_name, args.mutate_times)
    # # depart one model
    # if (test.res_model_dir / test.template_file_name).exists():
    #     print(args.model_name + " decomposition files exist.")
    #     pass
    # else:
    #     print(args.model_name + " decomposition file does not exist, we will create it……")
    #     test.depart()
    #     print(args.model_name + " decomposition complete.")


    # generate new model list
    test = MoCoTF("lenet", 2)
    if (test.res_model_dir / test.template_file_name).exists():
        print(test.model_name + " decomposition files exist.")
    else:
        print(test.model_name + " decomposition file does not exist, we will create it……")
        test.depart()
        print(test.model_name + " decomposition complete.")
    test.generate_model()

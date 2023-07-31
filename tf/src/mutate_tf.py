from pathlib import Path
import sys
# sys.path.append(Path.cwd().parent.parent.__str__())

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import random
import re
import tensorflow as tf
import time
import yaml
import traceback
from importlib import import_module
from math import ceil
from alive_progress import alive_bar
from queue import Queue

from tf.config.paths import RES_PATH, PARAM_PATH, FUNC_SIM_PATH, LOG_PATH, TF_MODEL_PATH, TF_PATH
from tf.config.models import models
from tf.src.logger_process_tf import log_process
from utils.MoCo import MoCo


class MoCoTF(MoCo):
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

    def __init__(self, model_name, mutate_times, is_mutate):
        super().__init__(model_name, mutate_times, is_mutate)
        self.model_name = model_name
        self.res_model_dir = RES_PATH / model_name
        self.log_model_dir = LOG_PATH / model_name
        self.time = int(time.time()).__str__()
        self.ERROR_NUM = 0

        self.mutate_dir = self.res_model_dir / ("mutate" + self.time)
        self.log_dir = self.log_model_dir / ("log" + self.time)

        self.ITERATION = 0
        self.MUTATE_TIMES = mutate_times
        self.Inception = {}

        self.template_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_template.py"
        self.function_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_function.py"
        self.inception_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_inception.py"

        if not Path.exists(RES_PATH):
            Path.mkdir(RES_PATH)

        if not Path.exists(self.res_model_dir):
            Path.mkdir(self.res_model_dir)

        if not Path.exists(LOG_PATH):
            Path.mkdir(LOG_PATH)

        if not Path.exists(self.log_model_dir):
            Path.mkdir(self.log_model_dir)

        with Path.open(TF_PATH / "data/api_list.txt", "r") as file:
            self.api_list = [_[3:-1] for _ in file]

        self.queue = Queue()
        self.queue.put(self.template_file_name)

        self.mutate_list = [self.mutate_on_param, self.mutate_on_function]
        self.detail_dict = {}

        self.NODE_ALIVE = 1
        self.THIS_NODE_ALL = 1
        self.NODE_ALL = 0
        self.NODE_RES = 1

        self.success_list = []

        self.is_mutate = is_mutate

    def depart(self):
        """
        Split network model
        """
        template_file = Path.open(Path(self.template_file_name), "a+", encoding="utf8")
        function_file = Path.open(Path(self.function_file_name), "w", encoding="utf8")
        inception_file = Path.open(Path(self.inception_file_name), "w", encoding="utf8")

        with Path.open(TF_MODEL_PATH / (self.model_name + ".py"), "r", encoding="utf8") as file_org:
            for line in file_org:
                if line.find("input_tensor") >= 0:
                    template_file.write("# " + self.model_name + " input layer" + "\n")
                    template_file.write(line)
                    template_file.write("# " + self.model_name + " hidden layer" + "\n")
                    break
                template_file.write(line)

            function_file.write("# " + self.model_name + " function layer" + "\n")
            for line in file_org:
                if line.find("output_tensor") >= 0:
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

    def mutate(self):
        """
        Mutate network model
        """
        if not Path.exists(self.mutate_dir):
            Path.mkdir(self.mutate_dir)

        if not Path.exists(self.log_dir):
            Path.mkdir(self.log_dir)

        function_file = Path.open(Path(self.function_file_name), "r", encoding="utf8")

        for line in function_file:
            tmp_queue = Queue()
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
                self.detail_dict = {}
                self.generate_model(line)
                self.run_model(line)
                print("THIS_NODE_ALL: " + str(self.THIS_NODE_ALL), end=" ")
                print("NODE_ALIVE: " + str(self.NODE_ALIVE), end=" ")
                print("NODE_RES: " + str(self.NODE_RES), end=" ")
                print("NODE_ALL: " + str(self.NODE_ALL))

    def generate_model(self, line):
        """
        Generate a layer of network model
        @param line: A line of mutable API code in the seed model
        """
        new_line = []
        new_mut = []
        num = 0

        self.ITERATION += 1

        function = self.get_function(line)
        self.THIS_NODE_ALL = self.NODE_RES * self.MUTATE_TIMES
        self.NODE_ALL += self.THIS_NODE_ALL
        if function not in self.api_list:
            if function in self.Inception.keys():
                count = self.Inception[function]
                self.Inception[function] = count + 1
            else:
                self.Inception[function] = 1
            new_function = function + "_" + self.Inception[function].__str__()
            new_line.append(line.replace(function, new_function))
        else:
            for i in range(self.THIS_NODE_ALL):
                method = random.choice(self.mutate_list)
                _new_line, label = method(line)
                new_line.append(_new_line)
                new_mut.append(label)

        while not self.queue.empty():
            org_file_name = self.queue.get()
            with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                content = org_file.read()
            output_pos = content.find("# " + self.model_name + " output layer")

            for i in range(self.MUTATE_TIMES):
                num += 1
                new_file_name = self.mutate_dir.resolve().__str__() + "/" + self.ITERATION.__str__() + "_" + num.__str__() + ".py"

                if function not in self.api_list:
                    __name__pos = content.find("if __name__")
                    if output_pos != -1 and __name__pos != -1:
                        new_module = self.mutate_on_module(function, self.Inception[function])
                        new_content = content[:output_pos] + new_line[0] + content[
                                                                           output_pos: __name__pos] + new_module + content[
                                                                                                                   __name__pos:]
                        new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                        new_file.write(new_content)
                        new_file.close()
                        if function in self.detail_dict:
                            self.detail_dict[function][new_file_name] = 0
                        else:
                            self.detail_dict[function] = {new_file_name: 0}

                else:
                    if output_pos != -1:
                        new_content = content[:output_pos] + new_line[num - 1] + content[output_pos:]
                        new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                        new_file.write(new_content)
                        new_file.close()
                        if new_mut[num - 1] in self.detail_dict:
                            self.detail_dict[new_mut[num - 1]][new_file_name] = 0
                        else:
                            self.detail_dict[new_mut[num - 1]] = {new_file_name: 0}

    def run_model(self, line):
        """
        Run a layer of network models
        @param line: A line of mutable API code in the seed model, same as the input of function generate_model
        """
        self.NODE_ALIVE = 0
        self.queue = Queue()
        model_list = []

        for _ in self.detail_dict.items():
            for __ in _[1].keys():
                model_list.append(__)

        with alive_bar(self.THIS_NODE_ALL, force_tty=True, title=("ITERATION : " + self.ITERATION.__str__())) as bar:
            for _ in self.detail_dict.items():
                for __ in _[1].keys():
                    model_name = __
                    bar()
                    try:
                        module_name = '.'.join(model_name.replace("/", ".").split(".")[-6:-1])
                        module = import_module(module_name)
                        if self.is_mutate:
                            self.detail_dict[_[0]][__] = module.go()
                        else:
                            net = getattr(module, self.model_name)()
                            self.detail_dict[_[0]][__] = net.count_params()
                            del net
                        self.NODE_ALIVE += 1
                        self.queue.put(model_name)
                    except Exception:
                        self.ERROR_NUM += 1
                        with Path.open(self.log_dir / Path("error" + self.ERROR_NUM.__str__()), "w",
                                       encoding="utf8") as file:
                            file.write(traceback.format_exc())

        if self.queue.empty():
            exit("There are no regenerated nodes, rerun or check the log contents.")

        self.beam_search(line)

    def get_function(self, line: str) -> str:
        """
        Get the API function name for that line
        @param line: A line of mutable API code in the seed model
        @return: API function name
        """
        try:
            function = re.findall(r".*? = (.*?)\(.*?", line)[0]
        except:
            return line
        return function

    def get_params(self, line: str) -> dict:
        """
        Get the API parameters for that line of code and return them as a dictionary
        @param line: A line of mutable API code in the seed model
        @return: parameters dictionary
        """
        if line.find("[") >= 0:
            line = line.replace("[", "(").replace("]", ")")

        infos1 = re.findall(r".*?\((?P<param>.*?)\)\((?P<input>.*?)\)", line)

        if not infos1:
            params = line.split("(")[1][:-1] + ', '
        else:
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
        """
        Generate a new line of code based on the API function name and the argument list, which is very similar to the original code
        @param line: A line of mutable API code in the seed model
        @param params_dict: New parameters dictionary
        @return: A new line of code
        """
        new_params: str = ""
        for _ in params_dict:
            new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

        new_params = new_params[:-2]
        new_params = "(" + new_params + ")"

        org_params = re.findall(r".*?(\(.*?\))\(.*?", line, re.S)
        if not org_params:
            org_params = "(" + line.split("(")[1] + "\n"
        else:
            org_params = org_params[0]

        new_line = line.replace(org_params, new_params, 1)
        return new_line

    def mutate_on_param(self, line: str) -> (str, str):
        """
        Parameter mutation
        @param line: A line of mutable API code in the seed model
        @return: (A new line of code, A label of this line)
        """
        dict = self.get_params(line)
        function = self.get_function(line)
        func_file = PARAM_PATH / ("tf." + function + ".yaml")
        if func_file.exists():
            with Path.open(func_file, "r") as file:
                all_data = yaml.load(file, yaml.Loader)
                data = all_data["constraints"]
                params_dict = dict.copy()

                param = self.random_param(data) if len(list(data.keys())) > 1 else list(data.keys())[0]
                value, label = self.get_value(data[param])
                label = param + ": " + str(label)
                params_dict[param] = value
                new_line = self.generate_line(line, params_dict)
        else:
            new_line = line
        return new_line, label

    def random_param(self, data) -> str:
        """
        The parameters to be mutated are selected according to the roulette wheel method
        @param data: Parameters dictionary
        @return: A parameter for this mutation
        """
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

    def mutate_on_function(self, line: str) -> (str, str):
        """
        API function name mutation
        @param line: A line of mutable API code in the seed model
        @return: (A new line of code, A label of this line)
        """
        function = self.get_function(line)
        func_file = FUNC_SIM_PATH / ("tf." + function + ".yaml")

        if func_file.exists():
            dict = self.get_params(line)
            with Path.open(func_file, "r") as file:
                data = yaml.load(file, yaml.Loader)
                func_mut = self.random_function(data)
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
                                tmp_dict[_], label = self.get_value(data[_])
                else:
                    tmp_dict = dict.copy()

                line = line.replace(function, func_mut[3:])
                _line = self.generate_line(line, tmp_dict)

                new_line, label = self.mutate_on_param(_line)

        return new_line, func_mut[3:]

    def random_function(self, data) -> str:
        """
        The target function of the API function mutation
        @param data: Function similarity dictionary of the source API
        @return: A function for this mutation
        """
        sim_sum = sum(data.values())
        probabilities = {func: value / sim_sum for func, value in data.items()}
        cumulative_probabilities = {}
        cumulative_sum = 0

        for func_mut, probability in probabilities.items():
            cumulative_sum += probability
            cumulative_probabilities[func_mut] = cumulative_sum

        random_num = random.random()

        for func_mut, cumulative_probability in cumulative_probabilities.items():
            if cumulative_probability >= random_num:
                return func_mut

    def get_value(self, dic):
        """
        Get the values for the selected parameters
        @param dic: A dictionary of specific values for the selected parameters, which is different from the input dictionary for random_param
        @return: A valid value for the parameter is selected
        """
        value = ""
        label = ""
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "tf.string":
                if "enum" in dic:
                    value = random.choice((dic["enum"]))
                    label = value
                    value = '"' + value + '"'
                else:
                    value = dic["default"]
                    label = value
                    if value == "None":
                        pass
                    else:
                        value = '"' + value + '"'
            elif type == "tf.bool":
                value = random.choice([True, False])
                label = value
            elif type == "float":
                value = random.random().__str__()
                if value == 0:
                    label = "min"
                elif value == 1:
                    label = "max"
                else:
                    label = "legal"
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
                        if value == min_v:
                            label = "min"
                        elif value == max_v:
                            label = "max"
                        else:
                            label = "legal"
                    elif structure == "tuple":
                        value = tuple(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                        if min_v in value:
                            label = "min"
                        elif max_v in value:
                            label = "max"
                        else:
                            label = "legal"
                    elif structure == "list":
                        value = list(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                        if min_v in value:
                            label = "min"
                        elif max_v in value:
                            label = "max"
                        else:
                            label = "legal"
                    elif structure == "tuple_of_tuples":
                        value = tuple(tuple(random.randint(min_v, max_v) for _ in range(2)) for _ in
                                      range(dic["shape"]))
                        if any(min_v in subtuple for subtuple in value) and any(
                                max_v in subtuple for subtuple in value):
                            label = "min_max"
                        elif any(min_v in subtuple for subtuple in value) and not any(
                                max_v in subtuple for subtuple in value):
                            label = "min"
                        elif any(max_v in subtuple for subtuple in value) and not any(
                                min_v in subtuple for subtuple in value):
                            label = "max"
                        else:
                            label = "legal"
                else:
                    value = dic["default"]
                    label = value
        else:
            value = dic["default"]

        return value, label

    def mutate_on_module(self, function: str, number: int) -> str:
        """
        Special mutation is applied to complex models
        @param function: The function name of inception block
        @param number: The number of mutations for the inception block
        @return: Mutation of the entire inception block
        """
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
            if line == "\n":
                def_list.append(line)
            elif line.find("# reshape") >= 0:
                def_list.append(line)
                break
            else:
                func = self.get_function(line)
                if func in self.api_list and func_mut != func and count <= 2:
                    func_mut = func
                    method = random.choice(self.mutate_list)
                    new_line, label = method(line)
                    def_list.append(new_line)
                    count += 1
                else:
                    def_list.append(line)

        for line in inception_file:
            if line.find("def") >= 0:
                break
            def_list.append(line)

        new_module = "".join(def_list)
        return new_module

    def beam_search(self, line):
        """
        Beam search
        @param line: A line of mutable API code in the seed model
        """
        self.queue = Queue()
        self.NODE_RES = 0
        function = self.get_function(line)

        if function not in self.api_list:
            # inception
            for _ in self.detail_dict.items():
                for __ in _[1].items():
                    if __[1] == 0:
                        pass
                    else:
                        self.queue.put(__[0])
                        self.NODE_RES += 1
        else:
            for _ in self.detail_dict.items():
                sorted_dict = dict(sorted(_[1].items(), key=lambda x: x[1]), reversed=False)
                if self.ITERATION < 10:
                    alive = ceil(len(sorted_dict) / 3)
                else:
                    alive = ceil(len(sorted_dict) / 4)
                for __ in sorted_dict.items():
                    if __[1] == 0:
                        pass
                    elif alive == 0:
                        break
                    else:
                        self.queue.put(__[0])
                        self.NODE_RES += 1
                        alive -= 1

                        with open(self.res_model_dir / (self.model_name + "_success.txt"), "a+") as file:
                            file.write(str(__[0]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--run_all',
                        type=bool,
                        default=False,
                        required=False,
                        help="Whether to run all models")
    parser.add_argument('--model_name',
                        type=str,
                        default="lenet",
                        required=False,
                        help="Model name")
    parser.add_argument('--mutate_times',
                        type=int,
                        default=3,
                        required=False,
                        help="The number of mutations per node")
    parser.add_argument('--is_train',
                        action='store_true',
                        help="Model trained or not")

    args = parser.parse_args()

    args.run_all = False
    if args.run_all:
        for model in models:
            moco_tf = MoCoTF(model, 3, False)
            print("{} Start".format(moco_tf.model_name))
            if (moco_tf.res_model_dir / moco_tf.template_file_name).exists():
                pass
            else:
                moco_tf.depart()
            moco_tf.mutate()
            log_process(moco_tf.model_name)
    else:
        # moco_tf = MoCoTF(args.model_name, args.mutate_times, args.is_train)
        moco_tf = MoCoTF("lenet", 2, False)
        if (moco_tf.res_model_dir / moco_tf.template_file_name).exists():
            print("{} decomposition files exist.".format(moco_tf.model_name))
        else:
            print("{} decomposition file does not exist, we will create it……".format(moco_tf.model_name))
            moco_tf.depart()
            print("{} decomposition complete.".format(moco_tf.model_name))

        moco_tf.mutate()
        log_process(moco_tf.model_name)

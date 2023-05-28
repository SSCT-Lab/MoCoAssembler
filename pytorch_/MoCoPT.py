import copy
import datetime
import os
import queue
import random
import re
import subprocess
from typing import List

import yaml

from config import marker
import utils


class MoCoPT:

    def __init__(self, model_name: str):
        self.model_name = model_name

        # 可变构件，拆分成字典和列表作为输入
        self.init_input = {}
        self.forward_input = []

        # 不可变模板，第一部分、第四部分以及二三之间
        self.begin = ""
        self.end = ""
        self.forward_line = ""

        # 可变构件的变异后输出
        self.init_output = ""
        self.forward_output = ""

        # 缩进量
        self.init_space = 0
        self.forward_spaces = []

        # init函数中单个API被变异次数统计
        self.init_api_visited = {}

        # 当前循环的输出行
        self.init_sentence = ""
        self.forward_sentence = ""
        self.changed = False

        # 当前信息，包含行号，当前变量名，变异后变量名，当前类别，参数列表解析字典
        self.line_idx = 0
        self.self_name = ""
        self.new_self_name = ""
        self.category = ""
        self.original_param_kvs = {}

        # 参数值
        self.INF_VAL = 4
        self.MUTATE_TIMES = 2

        # 队列
        self.que = queue.Queue()

    def generate_model(self):
        self.depart()

        # 为重组准备目录
        if not os.path.exists(marker.MUTATED_MODEL_PATH):
            os.makedirs(marker.MUTATED_MODEL_PATH)
        target = os.path.join(marker.MUTATED_MODEL_PATH, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(target)
        err_path = os.path.join(target, "error.log")

        # queue里面保存init_output的信息
        self.que.put(self.init_output)

        # 遍历forward函数的核心内容列表
        for i in range(len(self.forward_input)):
            # 匹配当前行是否为"self.xxx = xxx"的形式
            match = re.search(r'self\.\w+(?=\()', self.forward_input[i])
            if self.forward_input[i].startswith('#') or self.forward_input[i] == '\n' or not match:
                forward_sentence = " " * self.forward_spaces[i] + self.forward_input[i]
                if self.forward_input[i] != '\n':
                    forward_sentence += "\n"
                self.forward_output += forward_sentence
                continue

            # 匹配到了，提取"self.xxx"部分,并到init函数的dict中寻找对应的"nn.xxx"
            self_name = match.group()
            new_self_name = self_name + "_" + str(self.init_api_visited[self_name])
            self.self_name, self.new_self_name, self.line_idx = self_name, new_self_name, i

            # 对于queue中的每一个值，加上变异后的新语句
            que_len = self.que.qsize()

            for k1 in range(que_len):
                init_output = self.que.get()
                for k2 in range(self.MUTATE_TIMES):
                    self.changed = False
                    self.mutate()

                    # 写入文件
                    file_name_mut = self.model_name + "_mutated" + "-" + str(i) + "-" + str(k1) + "-" + str(k2) + ".py"
                    file_path_mut = os.path.join(target, file_name_mut)
                    with open(file_path_mut, "w+", encoding="utf8") as file_mut:
                        file_mut.write(self.begin)
                        file_mut.write(init_output + self.init_sentence + "\n")
                        file_mut.write(self.forward_line)
                        file_mut.write(self.forward_output + self.forward_sentence)
                        file_mut.write(self.end)
                        file_mut.close()

                    self.que.put(init_output + self.init_sentence)

                    # instr = "python " + file_path_mut
                    # process = subprocess.Popen(instr, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    # output, error = process.communicate()
                    #
                    # if error:
                    #     err_info = file_name_mut + ":\n" + error.decode("utf-8") + "\n\n"
                    #     with open(err_path, "w+", encoding="utf8") as file_err:
                    #         file_err.write(err_info)
                    #         file_mut.close()
                    # else:
                    #     self.que.put(init_output + self.init_sentence)

                    self.init_sentence = ""

                    # 未改变则只变异一次
                    if not self.changed:
                        break

            # 无论怎样变异，最后forward函数中的内容不变
            self.forward_output += self.forward_sentence

    def depart(self):
        # Step0: 判断文件属于简单还是复杂
        file_lst = os.listdir(marker.SIMPLE_MODEL_PATH)
        if self.model_name + ".py" in file_lst:
            file_path_org = os.path.join(marker.SIMPLE_MODEL_PATH, self.model_name + ".py")
        else:
            file_path_org = os.path.join(marker.COMPLEX_MODEL_PATH, self.model_name + ".py")

        # Step1: 解析模型文件
        with open(file_path_org, "r", encoding="utf8") as file_org:
            temp_list = []

            for line in file_org:
                if line.find("super(") >= 0:
                    temp_list.append(line)
                    self.begin = ''.join(temp_list)
                    temp_list.clear()

                elif line.find("def forward(") >= 0:
                    self.forward_line = line
                    init_lines = copy.deepcopy(temp_list)
                    temp_list.clear()

                elif line.find("return x") >= 0:
                    forward_lines = copy.deepcopy(temp_list)
                    temp_list.clear()
                    temp_list.append(line)

                else:
                    temp_list.append(line)

            self.end = ''.join(temp_list)
            file_org.close()

        # Step2: 预处理第二部分(init函数)保存至字典中, 并计算缩进量
        for line in init_lines:
            if line.strip(' ') == '\n' or line.strip().startswith('#'):
                continue
            lst = line.strip().split("=", 1)
            self.init_space = utils.count_space(line)

            if "nn." in lst[1]:
                api_name = lst[0].strip()
                para_lst = lst[1].strip()
                self.init_input[api_name] = para_lst
                self.init_api_visited[api_name] = 0
            else:
                self.init_output += line

        # Step3: 预处理第三部分，计算forward函数每行的缩进量以及核心内容
        for line in forward_lines:
            self.forward_spaces.append(utils.count_space(line))
            if line.strip(' ') == '\n':
                self.forward_input.append("\n")
            else:
                self.forward_input.append(line.strip())

    def mutate(self):
        para_line = self.init_input[self.self_name]
        api_nn = self.get_function(para_line)
        api_torch = "torch." + api_nn
        para_lst = self.get_params(para_line)

        # 准备好输出的init行前半句，以及forward行全部
        init_sentence = " " * self.init_space + self.new_self_name + " = "
        new_forward_line = self.forward_input[self.line_idx].replace(self.self_name, self.new_self_name)
        forward_sentence = " " * self.forward_spaces[self.line_idx] + new_forward_line + "\n"

        # 得到当前"nn.xxx"所在的类别
        try:
            category = utils.get_category(api_torch)
        except Exception as err:
            init_sentence += para_line + "\n"
            self.init_sentence = init_sentence
            self.forward_sentence = forward_sentence
            print(err)
            return

        self.category = category

        # 获取constraints内容并解析para_lst
        self.original_param_kvs.clear()
        constraint_file = os.path.join(marker.CONSTRAINTS_PATH, category, api_torch + ".yaml")
        with open(constraint_file, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()
        param_names = content["inputs"]["required"] + content["inputs"]["optional"]
        for idx, para in enumerate(para_lst):
            if '=' in para:
                kv_pair = para.split('=')
                self.original_param_kvs[kv_pair[0].strip()] = kv_pair[1].strip()
            else:
                self.original_param_kvs[param_names[idx]] = para

        if utils.to_mutate_on_api(category):
            simi_path = os.path.join(marker.CONSTRAINTS_PATH, category, "similarity_avg.yaml")
            target_api = self.mutate_on_function(api_torch, simi_path)
            api_nn = target_api.split('.', 1)[1]
            constraint_file = os.path.join(marker.CONSTRAINTS_PATH, category, target_api + ".yaml")

        line = self.mutate_on_parma(init_sentence + api_nn + "(", constraint_file)
        self.init_sentence = line[:-2] + ")\n"
        self.forward_sentence = forward_sentence
        self.changed = True

    def mutate_on_function(self, line: str, func_file: str) -> str:
        with open(func_file, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()

        if self.category == "conv" or self.category == "pooling":
            target = utils.select_api_conv_pool(content[line], line)
        else:
            target = utils.select_api_by_probability(content[line])

        return target

    def mutate_on_parma(self, line: str, func_file: str) -> str:
        with open(func_file, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()

        constr, input_list = content["constraints"], content["inputs"]
        required_list = input_list["required"]
        for idx, item in enumerate(required_list):
            if "range" in constr[item].keys() and constr[item]["range"] == "LIMITED":
                val = self.original_param_kvs[item]
            else:
                val = self.generate_param_line("", constr[item])

            if self.category == "linear" and item == "in_features":
                self.forward_sentence = " " * self.forward_spaces[
                    self.line_idx] + "x = nn.AdaptiveAvgPool1d(" + val + ")(x)\n"

            line += item + "=" + str(val) + ", "

        opt_lst = input_list["optional"]
        for idx, item in enumerate(opt_lst):
            if random.random() < idx / len(opt_lst) or item in ["return_indices", "groups", "dilation", "padding_idx"]:
                continue

            val = -1 if item == "dim" else self.generate_param_line("", constr[item])
            line += item + "=" + str(val) + ", "

        return line

    def generate_param_line(self, line: str, params_dict: dict) -> str:
        if "dtype" not in params_dict.keys():
            return "None"

        default_val = params_dict["default"] if "default" in params_dict.keys() else "None"
        ran = params_dict["range"] if "range" in params_dict.keys() else "None"
        dtype = params_dict["dtype"]

        if len(dtype) == 1:
            if dtype[0].startswith("enum"):
                val = '"' + random.choice(ran) + '"'
            else:
                val = self.random_param([default_val, dtype[0], ran])
        else:
            type_chosen = random.choice(dtype)
            if type_chosen.startswith("Tuple"):
                inner_type = dtype[1 - dtype.index(type_chosen)]
                match = re.search(r'\((\d+)\)$', type_chosen)
                num = int(match.group(1))

                lst = []
                for _ in range(num):
                    rand_para = self.random_param([default_val, inner_type, ran])
                    if inner_type == "int":
                        lst.append(int(rand_para))
                    else:
                        lst.append(float(rand_para))
                val = tuple(lst)
            else:
                val = self.random_param([default_val, type_chosen, ran])

        return val

    def random_param(self, data) -> str:
        [default, dtype, ran] = data
        val = "None"

        if ran == "None":
            low, high = 1, self.INF_VAL
        else:
            match1 = re.match(r'\[(.*),\s*(.*)]', str(ran))
            match2 = re.match(r'\[(.*),\s*(.*)\)', str(ran))
            if match1:
                low, high = match1.group(1), match1.group(2)
            elif match2:
                low, high = match2.group(1), match2.group(2)
            else:
                low, high = 1, self.INF_VAL

            if high == "inf":
                high = self.INF_VAL
            elif high == "ks/2":
                ks = self.original_param_kvs["kernel_size"]
                if ',' in ks:
                    match = re.match(r'\((\d+), (\d+)\)', ks)
                    high = min(int(match.group(1)), int(match.group(2))) // 2
                else:
                    high = int(ks) // 2
            elif high == "ne":
                high = 1

        if dtype == "int":
            if default == "None":
                val = random.randint(1, self.INF_VAL)
            else:
                val = utils.random_normal_int(int(default), low, high)
        elif dtype == "torch.bool":
            val = random.randint(0, 1) == 1
        elif dtype == "torch.float32":
            if default == "None":
                val = random.random() * self.INF_VAL
            else:
                val = utils.random_normal_float(float(default), low, high)

        return str(val)

    def get_function(self, line: str) -> str:
        return line.split('(', 1)[0]

    def get_params(self, line: str) -> List[str]:
        # TODO:解析list的bug
        return line.split('(', 1)[1].split(')', -1)[0].split(',')

    def mutate_on_module(self, function: str, Inception: dict) -> str:
        pass


if __name__ == "__main__":
    m = MoCoPT("demo")
    m.generate_model()

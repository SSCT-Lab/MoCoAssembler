import queue
import random
import re
import subprocess
from pathlib import Path
from queue import Queue

import yaml

from config.paths import RES_PATH, LOG_PATH, FUNC_PATH, PT_MODEL_PATH, FUNC_SIM_PATH, PARAM_PATH, PATH
from utils.MoCo import MoCo


class MoCoPT(MoCo):
    def __init__(self, model_name: str, mutate_times):
        super().__init__(model_name, mutate_times)
        self.MUTATE_TIMES = 2
        self.ITERATION = 0
        self.model_name = model_name
        self.res_model_dir = RES_PATH / model_name
        self.mutate_dir = self.res_model_dir / "mutate"
        self.log_file = LOG_PATH

        self.iteration = 0
        self.mutate_times = mutate_times

        self.template_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_template.py"
        self.init_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_init.py"
        self.forward_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_forward.py"
        self.inception_file_name = self.res_model_dir.resolve().__str__() + "/" + self.model_name + "_inception.py"

        if not Path.exists(RES_PATH):
            Path.mkdir(RES_PATH)

        if not Path.exists(self.res_model_dir):
            Path.mkdir(self.res_model_dir)

        if not Path.exists(self.mutate_dir):
            Path.mkdir(self.mutate_dir)

        with Path.open(PATH / "data/api_list", "r") as file:
            self.api_list = [_[:-1] for _ in file]

        self.queue = Queue()
        self.queue.put(self.template_file_name)

        self.mutate_list = [self.mutate_on_param, self.mutate_on_function]
        self.error_list = []

    def depart(self):
        template_file = Path.open(Path(self.template_file_name), "a+", encoding="utf8")
        init_file = Path.open(Path(self.init_file_name), "w", encoding="utf8")
        forward_file = Path.open(Path(self.forward_file_name), "w", encoding="utf8")
        inception_file = Path.open(Path(self.inception_file_name), "w", encoding="utf8")

        with Path.open(PT_MODEL_PATH / (self.model_name + ".py"), "r", encoding="utf8") as file_org:
            for line in file_org:
                if line.find("super(" + self.model_name) >= 0:
                    template_file.write(line)
                    template_file.write("# " + self.model_name + " forward layer\n\n")
                    break
                template_file.write(line)

            for line in file_org:
                if line.find("def") >= 0:
                    template_file.write(line)
                    template_file.write("# " + self.model_name + " output layer\n\n")
                    break
                init_file.write(line)
            init_file.close()

            for line in file_org:
                if line.find("return") >= 0:
                    template_file.write(line)
                    break
                forward_file.write(line)
            forward_file.close()

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

        forward_file = Path.open(Path(self.forward_file_name), "r", encoding="utf8")

        init_line = ''
        for forward_line in forward_file:
            if forward_line.strip().startswith("#") or forward_line == "\n":
                continue
            elif forward_line.find("view") >= 0:
                while not self.queue.empty():
                    org_file_name = self.queue.get()
                    with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                        content = org_file.read()
                        forward_pos = content.find("# " + self.model_name + " output layer")
                        if forward_pos != -1:
                            new_content = content[:forward_pos] + forward_line + content[forward_pos:]
                    with Path.open(Path(org_file_name), "w", encoding="utf8") as file:
                            file.write(new_content)
                    tmp_queue.put(org_file_name)

                while not tmp_queue.empty():
                    self.queue.put(tmp_queue.get())
                continue

            else:
                self.ITERATION += 1

                forward_function = self.get_function(forward_line)
                print(forward_function)
                with Path.open(Path(self.init_file_name), "r", encoding="utf8") as init_file:
                    for init_line in init_file:
                        if init_line.find(forward_function) >= 0:
                            break
                function = self.get_function(init_line)
                if function not in self.api_list:
                    if function in Inception.keys():
                        count = Inception[function]
                        Inception[function] = count + 1
                    else:
                        Inception[function] = 1
                    new_function = function + "_" + Inception[function].__str__()
                    new_line.append(init_line.replace(function, new_function))

                else:
                    for i in range(pow(self.MUTATE_TIMES, self.ITERATION)):
                        method = random.choice(self.mutate_list)
                        new_line.append(method(init_line))
                        # new_line.append(init_line)

            while not self.queue.empty():
                org_file_name = self.queue.get()
                with Path.open(Path(org_file_name), "r", encoding="utf8") as org_file:
                    content = org_file.read()
                    init_pos = content.find("# " + self.model_name + " forward layer")
                    forward_pos = content.find("# " + self.model_name + " output layer")

                    for i in range(1, self.MUTATE_TIMES + 1):
                        num += 1
                        new_file_name = self.mutate_dir.resolve().__str__() + "/" + self.ITERATION.__str__() + "_" + num.__str__() + ".py"
                        tmp_queue.put(new_file_name)

                        if function not in self.api_list:
                            inception_pos = content.find("if __name__")
                            if init_pos != -1 and forward_pos != -1 and inception_pos != -1:
                                new_module = self.mutate_on_module(function, Inception[function])
                                new_content = content[:init_pos] + new_line[0] + content[init_pos:forward_pos] + forward_line + content[forward_pos: inception_pos] + new_module + content[inception_pos:]
                                new_file = Path.open(Path(new_file_name), "w", encoding="utf8")
                                new_file.write(new_content)
                                new_file.close()

                        else:
                            if init_pos != -1 and forward_pos != -1:
                                new_content = content[:init_pos] + new_line[num - 1] + content[init_pos:forward_pos] + forward_line + content[forward_pos:]
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

            # 仅生成代码，不考虑运行结果
            # while not tmp_queue.empty():
            #     model = tmp_queue.get()
            #     self.queue.put(model)

            num = 0
            new_line = []
            if self.queue.empty():
                break


    def get_function(self, line: str) -> str:
        function = re.findall(r".*? = (.*?)\(.*?", line)[0]
        return function

    def get_params(self, line: str) -> dict:
        infos1 = re.findall(r".*?\((?P<param>.*?)\)", line)

        params = infos1[0] + ', '

        infos2 = re.findall(r".*?\((.*?)\).*?", params, re.S)
        for _ in infos2:
            __ = _.replace(" ", "")
            params = params.replace(_, __)

        params_dic = {}
        details = re.finditer(r"(?P<param>.*?)=(?P<value>.*?), ", params, re.S)
        for _ in details:
            params_dic.update({_["param"].strip(): _["value"].strip()})

        return params_dic

    def generate_line(self, line: str, params_dict: dict) -> str:
        new_params: str = ""
        for _ in params_dict:
            new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

        new_params = new_params[:-2]
        new_params = "(" + new_params + ")"

        org_params: str = re.findall(r".*?(\(.*?\))", line, re.S)[0]

        new_line = line.replace(org_params, new_params, 1)
        return new_line

    def mutate_on_param(self, line: str) -> str:
        dict = self.get_params(line)
        function = self.get_function(line)
        func_file = PARAM_PATH / ("torch." + function + ".yaml")
        if func_file.exists():
            with Path.open(func_file, "r") as file:
                all_data = yaml.load(file, yaml.Loader)
                data = all_data["constraints"]
                params_dict = dict.copy()
                # 随机选择一个参数进行变异
                param = random.choice(list(data.keys())) if len(list(data.keys())) > 1 else list(data.keys())[0]
                value = self.get_value(data[param])
                params_dict[param] = value
                new_line = self.generate_line(line, params_dict)
        else:
            new_line = line
        return new_line

    def mutate_on_function(self, line: str) -> str:
        function = self.get_function(line)
        func_file = FUNC_SIM_PATH / ("torch." + function + ".yaml")

        # 相似度阈值
        th = 0.6
        if func_file.exists():
            dict = self.get_params(line)
            with Path.open(func_file, "r", encoding="utf8") as file:
                data = yaml.load(file, yaml.Loader)
                lst = [_[0] for _ in data.items() if _[1] > th]
                func_mut = random.choice(lst[1:]) if len(lst) > 1 else lst[0]
                param_file = PARAM_PATH / (func_mut + ".yaml")
                tmp_dict = {}
                if param_file.exists():
                    with Path.open(PARAM_PATH / (func_mut + ".yaml"), "r") as param_file:
                        data = yaml.load(param_file, yaml.Loader)
                        params_dict = data["constraints"]
                        required_list = data['inputs']["required"]
                        for _ in dict.items():
                            if _[0] in params_dict.keys():
                                tmp_dict[_[0]] = _[1]

                        for _ in required_list:
                            if _ not in tmp_dict:
                                tmp_dict[_] = self.get_value(data[_])
                else:
                    # 有的函数的参数列表取值可能没有储存，所以直接copy，不改变其参数列表（可能会出错）
                    tmp_dict = dict.copy()

                line = line.replace(function, func_mut[6:])
                new_line = self.generate_line(line, tmp_dict)

            return new_line

    def get_value(self, dic):
        value = ""
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "enum[string]":
                if "range" in dic:
                    value = random.choice((dic["range"]))
                    value = '"' + value + '"'
                else:
                    value = dic["default"]
                    if value == "None":
                        pass
                    else:
                        value = '"' + value + '"'
            elif type == "torch.bool":
                value = random.choice([True, False])
            elif type == "float":
                value = random.random().__str__()
            elif type == "int":
                if "structure" in dic and "range" in dic:
                    structure = dic["structure"]
                    structure = random.choice(dic["structure"]) if isinstance(structure,
                                                                                      list) else structure
                    drange = dic["range"]
                    min_v = int(drange[0])
                    max_v = int(drange[1])

                    if structure == "integer":
                        value = random.randint(min_v, max_v)
                    elif structure == "Tuple[int](2)":
                        value = tuple(random.randint(min_v, max_v) for _ in range(2))
                    elif structure == "Tuple[int](3)":
                        value = tuple(random.randint(min_v, max_v) for _ in range(3))
                else:
                    value = dic["default"]
        else:
            value = dic["default"]

        return value

    def mutate_on_module(self, function: str, number: int) -> str:
        def_list = []

        inception_file = Path.open(Path(self.inception_file_name), "r", encoding="utf8")

        new_function = function + "_" + number.__str__()
        # module 不变化
        for line in inception_file:
            if line.startswith("class " + function):
                def_list.append(line.replace(function, new_function))
                break

        for line in inception_file:
            if line.strip().startswith("super"):
                def_list.append(line.replace(function, new_function))
            elif line.find("return") >= 0:
                def_list.append(line)
                break
            else:
                def_list.append(line)

        new_module = "".join(def_list)
        return new_module

    def test(self):
        function = "Inception"
        number = 2

        print(self.mutate_on_module(function, number))


if __name__ == "__main__":
    test = MoCoPT("lenet", 2)
    # test.depart()
    # test.test()
    test.generate_model()
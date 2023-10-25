import os.path as p
import random
import time
import yaml
import os
from abc import ABC, abstractmethod
from _database import database
import sys
from importlib import import_module
import traceback


def generate_line(api_name: str, params_dict: dict) -> str:
    new_params: str = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    return api_name + new_params


class Performer(ABC):
    def __init__(self):
        self.at = 1
        return

    @abstractmethod
    def get_library_name(self) -> str:
        pass

    @abstractmethod
    def translate(self, abstract_model: dict) -> str:
        # Given a dict of abstract model, translate it into a str which can be written into a .py file
        # And then this .py file will be a whole model.
        pass

    @abstractmethod
    def get_model_from_file(self, case_path: str, file_name: str):
        # Given a file path of a .py file, turn the net in this file to a model in memory.
        # Return this model.
        pass

    @abstractmethod
    def train(self, model) -> float:
        # Given a model, train it, and return time cost(if failed, return -1).
        pass

    @abstractmethod
    def run(self, model) -> (float, list[int], str):
        # Given a model, give a tensor and run it, then
        # 1. return time cost(-1 if failed).
        # # # 2. return the calculate result(-1 if failed). (MODE 1 DONT RETURN)
        # 3. return the shape([] if failed).
        # 4. return the error info("" if succeeded).
        pass


class TorchPerformer(Performer):
    # TODO
    def __init__(self):
        super().__init__()
        import torch
        # import Trainers.TorchTrainer as Trainer
        self.LeNet_test_tensor = torch.randn(3, 1, 28, 28)
        self.s244_test_tensor = torch.randn(3, 3, 244, 244)

        self.LeNet_test_code = "    x = torch.randn(3, 1, 28, 28)\n    y = model(x)\n    return model\n"
        self.s244_test_code = "    x = torch.randn(3, 3, 244, 244)\n    y = model(x)\n    return model\n"
        return

    def __get_test_code(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_code
        elif model_name == "googlenet":
            return self.s244_test_code
        else:
            return None

    def __get_test_tensor(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_tensor
        elif model_name == "googlenet":
            return self.s244_test_tensor
        else:
            return None

    def get_library_name(self) -> str:
        return "torch"

    def translate(self, abstract_model: dict) -> str:
        head = "import torch\nimport torch.nn as nn\n\n\n"
        body = ""
        model_name_list = list(abstract_model.keys())
        main_model_name = model_name_list[0]
        for model in abstract_model:
            body += self.__dict_to_model_class(abstract_model[model], model, model_name_list)
        return head + body + "def go():\n    model = " + main_model_name + "()\n" +\
            self.__get_test_code(main_model_name)

    def get_model_from_file(self, case_path: str, file_name: str):
        model_name = file_name.replace(".py", "")
        sys.path.append(case_path)
        model = import_module(model_name)
        try:
            model = model.go()
        except Exception:
            model = "error message: \n" + str(traceback.format_exc())
        sys.path.remove(case_path)
        return model

    def train(self, model) -> float:
        # fake train
        return random.choice([random.uniform(3.0, 100.0), random.uniform(100.0, 200.0), random.uniform(1.0, 3.0)])

    def run(self, model) -> (float, list[int], str):
        if isinstance(model, str):
            return -1.0, [], model
        start_time = time.time()
        model_name = str(model).split("(", 1)[0]
        test_tensor = self.__get_test_tensor(model_name)
        flag = True
        error_message = ""
        try:
            y = model(test_tensor)
            shape = list(y.shape)
        except Exception:
            flag = False
            shape = []
            error_message = str(traceback.format_exc())
        if flag:
            end_time = time.time()
            time_cost = end_time - start_time
        else:
            time_cost = -1.0

        return time_cost, shape, error_message

    def __dict_to_model_class(self, model_dict: dict, model_name: str, model_name_list: list[str]) -> str:
        def_part = "class " + model_name + "(nn.Module):\n    def __init__(self):\n        super("\
                   + model_name + ", self).__init__()\n"
        forward_part = "    def forward(self, x):\n"
        # layer_name = "layer"
        layer_index = 0
        if isinstance(model_dict[0], list):
            _model_dict = model_dict[1:]
            extra = ", ".join(model_dict[0])
            def_part = "class " + model_name + "(nn.Module):\n    def __init__(self, " + extra + ")" +\
                       ":\n        super("+ model_name + ", self).__init__()\n"
        else:
            _model_dict = model_dict
        for layer_dict in _model_dict:
            layer_index += 1
            abstract_layer_name = layer_dict["layer"]
            if abstract_layer_name == "cat":
                forward_part_line = "        " + layer_dict["out"] + " = torch.cat("
                if isinstance(layer_dict["in"], list):
                    forward_part_line += str(layer_dict["in"]).replace("'", "")
                else:
                    forward_part_line += layer_dict["in"]
                forward_part_line += ", dim=" + str(layer_dict["params"]["dims"]) + ")\n"
                forward_part += forward_part_line
            elif abstract_layer_name in model_name_list:
                implicit_layer_name = abstract_layer_name
                implicit_params = layer_dict["params"]
                def_part_line = "        self." + "layer" + str(layer_index) + " = " +\
                                generate_line(implicit_layer_name, implicit_params) + "\n"
                forward_part_line = "        " + layer_dict["out"] + " = " + "self.layer" + str(layer_index) + "(" +\
                                    layer_dict["in"] + ")\n"
                def_part += def_part_line
                forward_part += forward_part_line
            else:
                implicit_layer_name = database.get_implicit_api_name(self.get_library_name(), abstract_layer_name)
                implicit_params = {}
                abstract_params = layer_dict["params"]
                for abstract_param_name in abstract_params.keys():
                    implicit_param_name = database.get_implicit_para_name(self.get_library_name(),
                                                                          abstract_layer_name, abstract_param_name)
                    implicit_params[implicit_param_name] = abstract_params[abstract_param_name]
                def_part_line = "        self." + "layer" + str(layer_index) + " = " +\
                                generate_line(implicit_layer_name, implicit_params) + "\n"
                forward_part_line = "        " + layer_dict["out"] + " = " + "self.layer" + str(layer_index) + "(" +\
                                    layer_dict["in"] + ")\n"
                def_part += def_part_line
                forward_part += forward_part_line
        return def_part + "\n" + forward_part + "        return x\n\n\n"


class JittorPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "jittor"

    def translate(self, abstract_model: dict) -> str:
        return "jittor model code"

    def get_model_from_file(self, case_path: str, file_name: str):
        return "jittor model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, list[int], str):
        return 1.0, ""


class TensorFlowPerformer(Performer):
    def __init__(self):
        # 你可以看看我的TorchPerformer怎么写的，或者运行一下我的各种函数看看效果，数据库在这里就是一个database对象
        # 数据库怎么用我有个文档
        # 然后下面有段小测试，在配置里把0号框架改成tensorflow，就自动加载TensorFlowPerformer。
        # 这段小测试意思就是把LeNet种子拉出来能通，就算这个Performer做好了
        # 文件确实很长你忍一下
        super().__init__()
        return

    def get_library_name(self) -> str:
        # 这个别动
        return "tensorflow"

    def translate(self, abstract_model: dict) -> str:
        # 这里给你输入一个abstract_model，就是我们的网络结构，这个方法要输出一个字符串，
        # 这个字符串是根据我们的抽象网络结构翻译好的一个完整的模型，可以直接write到py文件里不会报错的
        # 注意是字符串
        return "tensorflow model code"

    def get_model_from_file(self, case_path: str, file_name: str):
        # 这里给你输入一个路径，一个文件名。路径就是模型py文件所在的文件夹，文件名就是py文件的文件名
        # 示例输入：case_path = "D:/pythonProject/MoCo_F2/result/experiment1/LeNet-0-1",
        #          file_name = "LeNet-0-1_tensorflow.py"
        # 然后要把这个模型加载到内存中，并返回这个模型。
        return "tensorflow model"

    def train(self, model) -> float:
        # 先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管
        # 先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管
        return 1.0

    def run(self, model) -> (float, list[int], str):
        # 这里的model传入的是一个内存中的模型
        # 用它该用的形状跑它，然后输出：
        # 1. 运行时间，float类型（如果运行失败就输出-1）
        # 2. 输出形状，一个列表，list[int]（如果运行失败就输出[]空队列）
        # 3. 报错信息，str类型（如果运行成功没有报错信息就输出""空字符串）
        return 1.0, ""


def translator_factory(library: str) -> Performer | None:
    if library == "torch":
        return TorchPerformer()
    elif library == "tensorflow":
        return TensorFlowPerformer()
    elif library == "jittor":
        return JittorPerformer()
    else:
        return None


class Concrete:
    def __init__(self):
        self.__experiment_id = "experiment" + str(int(time.time()))
        print("experiment id: " + self.__experiment_id)
        RESULT_PATH = p.join("..", "result")
        os.makedirs(p.join(RESULT_PATH, self.__experiment_id))
        self.RESULT_PATH = p.join(os.getcwd(), "..", "result", self.__experiment_id)
        self.__model_name = ""
        self.__library_list: list[str] = []
        self.__performer_list: list[Performer] = []
        self.refresh_config()
        self.set_model_name("default")
        return

    def set_model_name(self, model_name: str) -> None:
        self.__model_name = model_name

    def get_experiment_path(self) -> str:
        return p.join("..", "result", self.__experiment_id)

    def get_experiment_id(self) -> str:
        return self.__experiment_id

    def new_experiment(self):
        self.__experiment_id = "experiment" + str(int(time.time()))
        print("experiment id: " + self.__experiment_id)
        RESULT_PATH = p.join("..", "result")
        os.makedirs(p.join(RESULT_PATH, self.__experiment_id))
        self.RESULT_PATH = p.join(os.getcwd(), "..", "result", self.__experiment_id)

    def refresh_config(self) -> None:
        old_library_list = self.__library_list
        config_path = p.join("..", "config", "config.yaml")
        f = open(config_path, "r", encoding="utf-8")
        config = yaml.full_load(f)
        f.close()
        self.__library_list = config["LIBRARY_LIST"].values()
        if old_library_list == self.__library_list:
            return
        self.__performer_list.clear()
        for library in self.__library_list:
            translator = translator_factory(library)
            self.__performer_list.append(translator)
            print(library + " translator loaded.")
        return

    def perform(self, abstract_model: dict, gen: int, index: int) -> list[dict]:
        result = []
        self.mo_co_assemble(abstract_model, gen, index, library="abstract")
        for performer in self.__performer_list:
            model_code = performer.translate(abstract_model)
            case_path, file_name = self.mo_co_assemble(model_code, gen, index, performer.get_library_name())
            model = performer.get_model_from_file(case_path, file_name)
            run_time_cost, shape, error_message = performer.run(model)
            run_test = True if run_time_cost >= 0 else False
            if not run_test:
                train_test = False
                train_time_cost = -1
                run_time_cost = -1
            else:
                train_time_cost = performer.train(model)
                train_test = False if train_time_cost < 0 else True
            result_dict = {"run test": run_test, "train test": train_test,
                           "train time cost": train_time_cost, "run time cost": run_time_cost,
                           "case path": case_path + "\\" + file_name, "shape": shape,
                           "error message": error_message}
            result.append(result_dict)
        return result

    def mo_co_assemble(self, model_code: str | dict, gen: int, index: int, library: str) -> (str, str):
        case_id = self.__model_name + "-" + str(gen) + "-" + str(index)
        case_path = p.join(self.RESULT_PATH, case_id)
        if not p.exists(case_path):
            os.makedirs(case_path)
        if library != "abstract":
            file_name = case_id + "_" + library + ".py"
            file_path = p.join(case_path, file_name)
        else:
            file_name = case_id + "_abstract.yaml"
            file_path = p.join(case_path, file_name)
            f = open(file_path, "w", encoding="utf-8")
            yaml.dump(model_code, f)
            f.close()
            return file_path
        f = open(file_path, "w", encoding="utf-8")
        f.write(model_code)
        f.close()
        return case_path, file_name


concrete = Concrete()
seed = database.get_seed("LeNet")
concrete.set_model_name("testLeNet")
result = concrete.perform(seed, 0, 1)

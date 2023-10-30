from __future__ import annotations

import os.path as p
import random
import time

import numpy as np
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
        """
        The abstract model is translated into a concrete model of the corresponding framework
        @param abstract_model: (str)Dictionary of layers before translate
        @return: model(str): Translated model code
        """
        pass

    @abstractmethod
    def get_model_from_file(self, case_path: str, file_name: str):
        """
        Given a file path of a .py file, turn the net in this file to a model in memory.
        @param case_path: (str)The file path where the current mutation model is stored
        @param file_name: (str)Current mutation model filename
        @return: model(TensorFlow|PyTorch|Jittor)
        """
        pass

    @abstractmethod
    def train(self, model) -> float:
        """
        Model training
        @param model: (TensorFlow|PyTorch|Jittor)The model returned by the function get_model_from_file()
        @return:
        """
        pass

    @abstractmethod
    def run(self, model) -> (float, list[int], str):
        """
        Model Running
        @param model: (TensorFlow|PyTorch|Jittor)The model returned by the function get_model_from_file()
        @return: (time_cost(float):   (-1 if failed),
                  shape(list[int]):   ([] if failed),
                  error_message(str): ("" if succeeded))
        """
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
    model_name = ''

    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "tensorflow"

    def translate(self, abstract_model: dict) -> str:
        self.model_name = list(abstract_model.keys())[0]
        head = "import tensorflow as tf\n"
        body = ""
        model_name_list = list(abstract_model.keys())
        main_model_name = model_name_list[0]
        for model in abstract_model:
            body += self.__dict_to_model_class(abstract_model[model], model, model_name_list)

        go = f'def go():\n' \
             f'    model = {main_model_name}(input_shape={self.__get_shape().__str__()})\n' \
             f'    return model'

        code = f'{head}\n' \
               f'{body}\n' \
               f'{go}\n'
        return code

    def get_model_from_file(self, case_path: str, file_name: str):
        module_name = file_name.split('.')[0]
        sys.path.append(case_path)
        model = import_module(module_name)
        try:
            model = model.go()
        except Exception:
            model = "error message: \n" + str(traceback.format_exc())
        sys.path.remove(case_path)
        return model

    def train(self, model) -> float:
        # 先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管
        # 先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管先不管
        return 1.0

    def run(self, model) -> (float, list[int], str):
        if isinstance(model, str):
            return -1.0, [], model
        start_time = time.time()
        test_tensor = self.__get_test_tensor()
        shape = []
        error_message = ""
        try:
            result = model(test_tensor)
            shape = list(result.shape)
            end_time = time.time()
            time_cost = end_time - start_time
        except Exception:
            end_time = time.time()
            time_cost = end_time - start_time
            error_message = str(traceback.format_exc())
        return time_cost, shape, error_message

    def __dict_to_model_class(self, model_dict: list, model_name: str, model_name_list: list[str]) -> str:
        layer_index = 0
        branchs = []
        input_layers = ''
        hidden_layers = '    # hidden layers\n'
        output_layers = '    # output layers\n'
        reshape_layer = ''
        if isinstance(model_dict[0], list):
            # inception block defination
            def_params = 'x, '
            def_params = f'{def_params}{", ".join(model_dict[0][1:])}'
            input_layers = f'{input_layers}def {model_name} ({def_params}):\n'

            _model_dict = model_dict[1:]
        else:
            input_layers = f'{input_layers}def {model_name}(input_shape):\n' \
                           f'    # input layers\n' \
                           f'    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")\n' \
                           f'    x = input_tensor\n'
            _model_dict = model_dict

        for layer in _model_dict:
            layer_index += 1
            abstract_layer_name = layer['layer']
            if abstract_layer_name == 'cat':
                # inception block
                reshape_layer = f'    # reshape layer\n' \
                                f'    target_height = x.shape[1]\n' \
                                f'    target_width = x.shape[2]\n'
                cat_params = ', '.join(branchs)
                for branch in branchs:
                    reshape_layer = f'{reshape_layer}' \
                                    f'    {branch} = tf.keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))({branch})\n'

                reshape_layer = f'{reshape_layer}' \
                                f'    outputs = tf.keras.layers.concatenate([{cat_params}])\n'
            elif abstract_layer_name in model_name_list:
                implicit_layer_name = abstract_layer_name
                implicit_params = dict([(layer['in'], layer['in'])])
                abstract_params = layer['params']
                for abstract_param_name in abstract_params:
                    if abstract_param_name == 'in_channels':
                        pass
                    else:
                        implicit_param_name = abstract_param_name
                        implicit_params.update(dict([(implicit_param_name, abstract_params[abstract_param_name])]))
                output_ = layer['out']
                hidden_layers = f'{hidden_layers}' \
                                f'    {output_} = {generate_line(implicit_layer_name, implicit_params)}\n'
            else:
                implicit_layer_name = database.get_implicit_api_name(self.get_library_name(), abstract_layer_name)
                implicit_params = {}

                # 单元测试时 : abstract_params = self.__convert_to_tf(abstract_layer_name, layer['params'])
                abstract_params = layer['params']
                for abstract_param_name in abstract_params:
                    implicit_param_name = abstract_param_name
                    implicit_params[implicit_param_name] = abstract_params[abstract_param_name]
                input_ = str(layer['in']).replace("'", "") if isinstance(layer['in'], list) else layer['in']
                output_ = layer['out']
                if output_ not in branchs:
                    branchs.append(output_)
                hidden_layers = f'{hidden_layers}' \
                                f'    {output_} = {generate_line(implicit_layer_name, implicit_params)}({input_})\n'

        if isinstance(model_dict[0], list):
            output_layers = f'{output_layers}' \
                            f'    return outputs\n'
        else:
            output_layers = f'{output_layers}' \
                            f'    output_tensor = x\n' \
                            f'    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)\n' \
                            f'    return model\n'

        code = f'{input_layers}\n' \
               f'{hidden_layers}\n' \
               f'{reshape_layer}\n' \
               f'{output_layers}\n'

        return code

    def __convert_to_tf(self, abstract_layer_name: str, para_dict: dict) -> dict:
        res_para_dict = {}
        for param in para_dict:
            # TODO ： 接口返回值有误
            # implicit_param_name = database.get_implicit_para_name(self.get_library_name(), abstract_layer_name, param)
            implicit_param_name = self.__get_implicit_para_name(abstract_layer_name, param)
            if implicit_param_name != "None":
                res_para_dict[implicit_param_name] = para_dict[param]
            else:
                pass
        # modify padding
        if "padding" in para_dict.keys():
            res_para_dict["padding"] = '"valid"' if para_dict["padding"] == 0 else '"same"'
        return res_para_dict

    def __get_implicit_para_name(self, abstract_layer_name: str, param: str) -> str:
        abs_para_mapping_path = '/Users/wuduo/Documents/BioWork/DifferentialTestingofDLFrameworks/MoCoAssembler/MoCo_F2/database/abstract/abstract_para_name.yaml'
        with open(abs_para_mapping_path, 'r', encoding='utf8') as file:
            abs_para_dict = yaml.load(file, yaml.Loader)

        return abs_para_dict[abstract_layer_name][param]['tensorflow']
    def __get_shape(self) -> tuple:
        if self.model_name == "LeNet":
            return 28, 28, 1
        elif self.model_name == "googlenet":
            return 224, 224, 3
        else:
            return ()

    def __get_test_tensor(self):
        if self.model_name == "LeNet":
            return np.random.rand(3, 28, 28, 1)
        elif self.model_name == "googlenet":
            return np.random.rand(3, 224, 224, 3)
        else:
            return ""


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
# seed = database.get_seed("LeNet")
# concrete.set_model_name("testLeNet")
# result = concrete.perform(seed, 0, 1)

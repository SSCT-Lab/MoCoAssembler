from __future__ import annotations

import os.path as p
import random
import numpy as np
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
        self.s224_test_tensor = torch.randn(3, 3, 224, 224)
        self.s299_test_tensor = torch.randn(3, 3, 299, 299)
        self.pointnet_test_tensor = torch.randn(3, 3, 5)
        self.lstm_test_tensor = torch.randn(5, 1)

        self.LeNet_test_code = "    x = torch.randn(3, 1, 28, 28)\n    y = model(x)\n    return model\n"
        self.s244_test_code = "    x = torch.randn(3, 3, 244, 244)\n    y = model(x)\n    return model\n"
        self.s224_test_code = "    x = torch.randn(3, 3, 224, 224)\n    y = model(x)\n    return model\n"
        self.s299_test_code = "    x = torch.randn(3, 3, 299, 299)\n    y = model(x)\n    return model\n"
        self.pointnet_test_code = "    x = torch.randn(3, 3, 5)\n    y = model(x)\n    return model\n"
        self.lstm_test_code = "    x = torch.randn(5, 1)\n    y = model(x)\n    return model\n"

        from TorchTrainer import torch_trainer
        self.trainer = torch_trainer
        return

    def __get_test_code(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_code
        elif model_name in ["googlenet", "resnet18", "resnet50", "alexnet", "densenet",
                            "mobilenet", "vgg16", "vgg19", "xception"]:
            return self.s224_test_code
        elif model_name == "inceptionv3":
            return self.s224_test_code
        elif model_name == "squeezenet":
            return self.s244_test_code
        elif model_name == "pointnet":
            return self.pointnet_test_code
        elif model_name == "lstm":
            return self.lstm_test_code
        else:
            return self.s224_test_code

    def __get_test_tensor(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_tensor
        elif model_name in ["googlenet", "resnet18", "resnet50", "alexnet", "densenet",
                            "mobilenet", "vgg16", "vgg19", "xception"]:
            return self.s224_test_tensor
        elif model_name == "inceptionv3":
            return self.s224_test_tensor
        elif model_name == "squeezenet":
            return self.s244_test_tensor
        elif model_name == "pointnet":
            return self.pointnet_test_tensor
        elif model_name == "lstm":
            return self.lstm_test_tensor
        else:
            return self.s224_test_tensor

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
        try:
            model = import_module(model_name)
            model = model.go()
        except Exception:
            model = "error message: \n" + str(traceback.format_exc())
        sys.path.remove(case_path)
        return model

    def train(self, model) -> float:
        # fake train
        # return random.choice([random.uniform(3.0, 100.0), random.uniform(100.0, 200.0), random.uniform(1.0, 3.0)])
        # True Train
        if isinstance(model, str):
            return -1.0
        else:
            start_time = time.time()
            try:
                self.trainer.train(model, str(model).split("(", 1)[0])
                end_time = time.time()
            except Exception:
                end_time = start_time - 1.0
        import torch.cuda
        torch.cuda.empty_cache()
        return end_time - start_time


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

    def __dict_to_model_class(self, model_dict: list, model_name: str, model_name_list: list[str]) -> str:
        def_part = "class " + model_name + "(nn.Module):\n    def __init__(self):\n        super("\
                   + model_name + ", self).__init__()\n"
        forward_part = "    def forward(self, x):\n"
        # layer_name = "layer"
        layer_index = 0
        if isinstance(model_dict[0], list):
            _model_dict = model_dict[1:]
            extra = ", ".join(model_dict[0])
            def_part = "class " + model_name + "(nn.Module):\n    def __init__(self, " + extra + ")" +\
                       ":\n        super(" + model_name + ", self).__init__()\n"
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
            elif abstract_layer_name == "add":
                forward_part_line =\
                    ("        " + layer_dict["out"] + " = " + "+".join(layer_dict["in"].split(",")) + "\n")
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
                    if implicit_param_name != "None":
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
        import jittor
        jittor.flags.use_cuda = 1
        from JittorTrainer import jittor_trainer
        self.trainer = jittor_trainer
        # import Trainers.TorchTrainer as Trainer
        self.LeNet_test_tensor = jittor.randn(3, 1, 28, 28)
        self.s244_test_tensor = jittor.randn(3, 3, 244, 244)
        self.s224_test_tensor = jittor.randn(3, 3, 224, 224)
        self.s299_test_tensor = jittor.randn(3, 3, 299, 299)
        self.pointnet_test_tensor = jittor.randn(3, 3, 5)
        self.lstm_test_tensor = jittor.randn(5, 1)

        self.LeNet_test_code = "    x = jittor.randn(3, 1, 28, 28)\n    y = model(x)\n    return model\n"
        self.s244_test_code = "    x = jittor.randn(3, 3, 244, 244)\n    y = model(x)\n    return model\n"
        self.s224_test_code = "    x = jittor.randn(3, 3, 224, 224)\n    y = model(x)\n    return model\n"
        self.s299_test_code = "    x = jittor.randn(3, 3, 299, 299)\n    y = model(x)\n    return model\n"
        self.pointnet_test_code = "    x = jittor.randn(3, 3, 5)\n    y = model(x)\n    return model\n"
        self.lstm_test_code = "    x = jittor.randn(5, 1)\n    y = model(x)\n    return model\n"
        return

    def get_library_name(self) -> str:
        return "jittor"

    def __get_test_code(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_code
        elif model_name in ["googlenet", "resnet18", "resnet50", "alexnet", "densenet",
                            "mobilenet", "vgg16", "vgg19", "xception"]:
            return self.s224_test_code
        elif model_name == "inceptionv3":
            return self.s224_test_code
        elif model_name == "squeezenet":
            return self.s244_test_code
        elif model_name == "pointnet":
            return self.pointnet_test_code
        elif model_name == "lstm":
            return self.lstm_test_code
        else:
            return None

    def __get_test_tensor(self, model_name: str):
        if model_name == "LeNet":
            return self.LeNet_test_tensor
        elif model_name in ["googlenet", "resnet18", "resnet50", "alexnet", "densenet",
                            "mobilenet", "vgg16", "vgg19", "xception"]:
            return self.s224_test_tensor
        elif model_name == "inceptionv3":
            return self.s224_test_tensor
        elif model_name == "squeezenet":
            return self.s244_test_tensor
        elif model_name == "pointnet":
            return self.pointnet_test_tensor
        elif model_name == "lstm":
            return self.lstm_test_tensor
        else:
            return self.s224_test_tensor

    def translate(self, abstract_model: dict) -> str:
        head = "import jittor\nimport jittor.nn as nn\n\n\n"
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
        try:
            model = import_module(model_name)
            model = model.go()
        except Exception:
            model = "error message: \n" + str(traceback.format_exc())
        sys.path.remove(case_path)
        return model

    def train(self, model) -> float:
        # True Train
        if isinstance(model, str):
            return -1.0
        else:
            start_time = time.time()
            try:
                self.trainer.train(model, str(model).split("(", 1)[0])
                end_time = time.time()
            except Exception:
                end_time = start_time - 1.0
                report = "=====================\n"
                report += traceback.format_exc()
                report += "=====================\n"
                with open("../report/train_error.txt", "a", encoding="utf-8") as fr:
                    fr.write(report)
        return end_time - start_time

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

    def __dict_to_model_class(self, model_dict: list, model_name: str, model_name_list: list[str]) -> str:
        def_part = "class " + model_name + "(nn.Module):\n    def __init__(self):\n        super("\
                   + model_name + ", self).__init__()\n"
        forward_part = "    def execute(self, x):\n"
        # layer_name = "layer"
        layer_index = 0
        if isinstance(model_dict[0], list):
            _model_dict = model_dict[1:]
            extra = ", ".join(model_dict[0])
            def_part = "class " + model_name + "(nn.Module):\n    def __init__(self, " + extra + ")" +\
                       ":\n        super(" + model_name + ", self).__init__()\n"
        else:
            _model_dict = model_dict
        for layer_dict in _model_dict:
            layer_index += 1
            abstract_layer_name = layer_dict["layer"]
            if abstract_layer_name == "cat":
                forward_part_line = "        " + layer_dict["out"] + " = jittor.cat("
                if isinstance(layer_dict["in"], list):
                    forward_part_line += str(layer_dict["in"]).replace("'", "")
                else:
                    forward_part_line += layer_dict["in"]
                forward_part_line += ", dim=" + str(layer_dict["params"]["dims"]) + ")\n"
                forward_part += forward_part_line
            elif abstract_layer_name == "add":
                forward_part_line =\
                    ("        " + layer_dict["out"] + " = " + "+".join(layer_dict["in"].split(",")) + "\n")
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
                    if implicit_param_name != "None":
                        implicit_params[implicit_param_name] = abstract_params[abstract_param_name]
                def_part_line = "        self." + "layer" + str(layer_index) + " = " +\
                                generate_line(implicit_layer_name, implicit_params) + "\n"
                forward_part_line = "        " + layer_dict["out"] + " = " + "self.layer" + str(layer_index) + "(" +\
                                    layer_dict["in"] + ")\n"
                def_part += def_part_line
                forward_part += forward_part_line
        return def_part + "\n" + forward_part + "        return x\n\n\n"


class TensorFlowPerformer(Performer):
    model_name = ''

    def __init__(self):
        super().__init__()
        import TFTrainer
        self.mnist = TFTrainer.get_mnist()
        self.cifar10 = TFTrainer.get_cifar10()
        self.imagenet = TFTrainer.get_imagenet()
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
            model_body = self.__dict_to_model_class(abstract_model[model], model, model_name_list)
            if model in ["alexnet", "vgg16", "vgg19", "googlenet", "squeezenet", "mobilenet", "resnet18", "lstm", "pointnet"]:
                model_body = model_body.replace('    output_tensor = x',
                                                "    output_tensor = tf.keras.layers.Flatten()(tf.keras.layers.Dense(units=1000, activation='softmax')(x))")
            elif model in ["LeNet"]:
                model_body = model_body.replace('    output_tensor = x',
                                                "    output_tensor = tf.keras.layers.Flatten()(tf.keras.layers.Dense(units=10, activation='softmax')(x))")
            body += model_body

        go = f'def go():\n' \
             f'    model = {main_model_name}(input_shape={self.__get_shape().__str__()})\n' \
             f'    x = tf.random.normal(shape=(1,) + {self.__get_shape().__str__()})\n' \
             f'    y = model(x)\n' \
             f'    return model\n'

        code = f'{head}\n' \
               f'{body}\n' \
               f'{go}\n\n'
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
        import tensorflow as tf
        model_name = self.model_name
        if model_name == "LeNet":
            x_train, y_train, x_test, y_test = self.mnist
        elif model_name in ["alexnet", "vgg16", "vgg19"]:
            x_train, y_train, x_test, y_test = self.cifar10
        elif model_name in ["pointnet", "lstm"]:
            return 1.0
        else:
            x_train, y_train, x_test, y_test = self.imagenet
        start_time = time.time()
        try:
            model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
            model.evaluate(x_test, y_test, verbose=0)
            end_time = time.time()
        except Exception:
            train_errors = self.get_library_name() + "  " + self.model_name + "\n"
            train_errors += traceback.format_exc() + "\n============\n"
            with open("../report/train_errors.txt", "a", encoding="utf-8") as fre:
                fre.write(train_errors)
            end_time = start_time - 1.0
        cost = end_time - start_time
        return cost

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
        input_layers = ''
        hidden_layers = '    # hidden layers\n'
        output_layers = '    # output layers\n'
        if isinstance(model_dict[0], list):
            # inception block defination
            def_params = 'x, '
            def_params = f'{def_params}{", ".join(model_dict[0])}'
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
            if abstract_layer_name in ['cat', 'add']:
                reshape_layer = f'    # reshape layer\n' \
                                f'    target_height = x.shape[1]\n' \
                                f'    target_width = x.shape[2]\n'
                branchs = layer['in']
                branchs = branchs if isinstance(branchs, list) else branchs.split(', ')
                cat_add_params = ', '.join(branchs)
                for branch in branchs:
                    reshape_layer = f'{reshape_layer}' \
                                    f'    {branch} = tf.keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))({branch})\n'
                if abstract_layer_name == 'cat':
                    hidden_layers = f'{hidden_layers}' \
                                    f'{reshape_layer}' \
                                    f'    {layer["out"]} = tf.keras.layers.concatenate([{cat_add_params}])\n'
                else:
                    hidden_layers = f'{hidden_layers}' \
                                    f'{reshape_layer}' \
                                    f'    {layer["out"]} = tf.keras.layers.add([{cat_add_params}])\n'
            elif abstract_layer_name in model_name_list:
                implicit_layer_name = abstract_layer_name
                implicit_params = dict([(layer['in'], layer['in'])])
                abstract_params = layer['params']
                for abstract_param_name in abstract_params:
                    implicit_param_name = abstract_param_name
                    implicit_params.update(dict([(implicit_param_name, abstract_params[abstract_param_name])]))
                output_ = layer['out']
                hidden_layers = f'{hidden_layers}' \
                                f'    {output_} = {generate_line(implicit_layer_name, implicit_params)}\n'
            else:
                # abstract_params = self.__convert_to_tf(abstract_layer_name, layer['params'])
                abstract_params = layer["params"]
                implicit_layer_name = database.get_implicit_api_name(self.get_library_name(), abstract_layer_name)
                implicit_params = {}
                for abstract_param_name in abstract_params:
                    implicit_param_name = abstract_param_name
                    implicit_params[implicit_param_name] = abstract_params[abstract_param_name]
                input_ = str(layer['in']).replace("'", "") if isinstance(layer['in'], list) else layer['in']
                output_ = layer['out']
                hidden_layers = f'{hidden_layers}' \
                                f'    {output_} = {generate_line(implicit_layer_name, implicit_params)}({input_})\n'

        if isinstance(model_dict[0], list):
            output_layers = f'{output_layers}' \
                            f'    return x\n'
        else:
            output_layers = f'{output_layers}' \
                            f'    output_tensor = x\n' \
                            f'    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)\n' \
                            f'    return model\n'

        code = f'{input_layers}\n' \
               f'{hidden_layers}\n' \
               f'{output_layers}\n'

        return code

    def __convert_to_tf(self, abstract_layer_name: str, para_dict: dict) -> dict:
        """
        Convert an abstract parameter name to a corresponding parameter name
        @param para_dict: Abstract argument list
        @return: Transformed argument list
        """
        res_para_dict = {}

        for param in para_dict:
            implicit_param_name = database.get_implicit_para_name(self.get_library_name(), abstract_layer_name, param)
            if implicit_param_name != "None":
                res_para_dict[implicit_param_name] = para_dict[param]
            else:
                pass
        # modify padding
        if "padding" in para_dict.keys():
            res_para_dict["padding"] = '"valid"' if para_dict["padding"] == 0 else '"same"'
        return res_para_dict

    def __get_shape(self) -> tuple:
        if self.model_name == "LeNet":
            return 28, 28, 1
        else:
            return 224, 224, 3

    def __get_test_tensor(self):
        if self.model_name == "LeNet":
            return np.random.rand(3, 28, 28, 1)
        else:
            return np.random.rand(3, 224, 224, 3)


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
                if gen <= 3:
                    train_time_cost = 1.0
                else:
                    try:
                        train_time_cost = performer.train(model)
                    except Exception:
                        train_time_cost = -1
                        report = "=====================\n" + "from: " + file_name + "\n"
                        report += traceback.format_exc()
                        report + "=====================\n"
                        with open("../report/train_error.txt", "a", encoding="utf-8") as fr:
                            fr.write(report)
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
if __name__ == "__main__":
    performer = TensorFlowPerformer()
    # net_list = ["alexnet", "LeNet", "mobilenet", "squeezenet", "vgg16",
    #             "vgg19", "googlenet", "resnet18", "pointnet", "lstm"]
    #
    #
    # def test(net: str):
    #     time.sleep(1.0)
    #     concrete.new_experiment()
    #     seed = database.get_seed(net)
    #     concrete.set_model_name("test_" + net)
    #     result = concrete.perform(seed, 0, 1)
    #     if not result[0]["run test"]:
    #         print(net + " has some questions")
    #     return result[0]

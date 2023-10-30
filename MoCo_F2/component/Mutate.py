from __future__ import annotations

import copy
from abc import ABC, abstractmethod
import random

import yaml

from _database import database


def roulette_wheel_selection(dictionary):
    total_value = sum(dictionary.values())
    random_number = random.uniform(0, total_value)
    current_sum = 0
    for key, value in dictionary.items():
        current_sum += value
        if current_sum >= random_number:
            return key


class Mutator(ABC):
    def __init__(self):
        return

    @abstractmethod
    def mutate(self, layer_dict: dict) -> (dict, str):
        """
        The main function of the variation on a single layer
        @param layer_dict: (dict)Dictionary of layers before mutation
        @return: (result_layer_dict: (dict)Dictionary of layers after mutation,
                  res_mutate_type: (str)A label of this mutate)
        """
        # 随机选择api替换和参数替换，然后返回变异后的新layer_dict以及变异信息（用于剪枝）
        # 注意，此处传入的都是针对抽象层面的layer_dict
        pass

    @abstractmethod
    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        """
        API function name mutation
        @param layer_dict: (dict)Dictionary of layers before mutation
        @return: (result_layer_dict: (dict)Dictionary of layers after mutation,
                  res_mutate_type: (str)A label of this mutate)
        """
        pass

    @abstractmethod
    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        """
        Parameter mutation
        @param layer_dict: (dict)Dictionary of layers before mutation
        @return: (result_layer_dict: (dict)Dictionary of layers after mutation,
                  res_mutate_type: (str)A label of this mutate)
        """
        pass

    @abstractmethod
    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        """
        Special mutation is applied to complex models
        @param layer_dict: (dict)Dictionary of layers before mutation
        @return: (result_layer_dict: (dict)Dictionary of layers after mutation,
                  res_mutate_type: (str)A label of this mutate)
        """
        # 当进行子模型变异时，传入的不是一层而是一个子模型了
        pass


class TorchMutator(Mutator):
    def __init__(self):
        super().__init__()
        self.count = 0

    def mutate(self, layer_dict: dict) -> (dict, str):
        if "layer" in layer_dict.keys():
            abstract_layer_name = layer_dict["layer"]
        else:
            abstract_layer_name = list(layer_dict.keys())[0]
        if abstract_layer_name == "cat":
            return layer_dict, "dont mutate this one"
        if not database.is_abstract_api_name_valid(abstract_layer_name):
            return self.child_model_mutate(layer_dict)
        if random.choice([1, 2]) == 1:
            return self.api_name_mutate(layer_dict)
        else:
            return self.api_para_mutate(layer_dict)

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        abstract_layer_name = layer_dict["layer"]
        implicit_layer_name = database.get_implicit_api_name("torch", abstract_layer_name)
        valid_similarity_dict = database.get_implicit_api_similarity_valid("torch", implicit_layer_name)
        if len(valid_similarity_dict) == 0:
            return layer_dict, "no mutate"
        else:
            new_implicit_layer_name = roulette_wheel_selection(valid_similarity_dict)
            new_abstract_layer_name = database.get_abstract_api_name("torch", new_implicit_layer_name)
            abstract_layer_info = database.get_abstract_layer_info(new_abstract_layer_name)
            required_list = abstract_layer_info["inputs"]["required"]
            param_constraints = abstract_layer_info["constraints"]
            new_para_dict = {}
            old_para_dict = layer_dict["params"]
            for p in old_para_dict.items():
                if p[0] in param_constraints.keys():
                    new_para_dict[p[0]] = p[1]
            for param_name in required_list:
                if param_name not in new_para_dict.keys():
                    new_para_dict[param_name] = self.__get_value(param_constraints[param_name])[0]
            return {"layer": new_abstract_layer_name, "params": new_para_dict,
                    "in": layer_dict["in"], "out": layer_dict["out"]}, new_abstract_layer_name

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        now_api_para_data = database.get_abstract_layer_info(layer_dict["layer"])["constraints"]
        params = copy.deepcopy(layer_dict["params"])
        choice_list = list(params.keys())
        no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                          'num_features']
        for no_mutate_para in no_mutate_pool:
            if no_mutate_para in choice_list:
                choice_list.remove(no_mutate_para)
        if len(choice_list) == 0:
            return layer_dict, 'no mutate'
        param_to_mutate = random.choice(choice_list)
        res_mutate_type = str(param_to_mutate)
        value, choice_type = self.__get_value(now_api_para_data[param_to_mutate])
        params[param_to_mutate] = value
        res_mutate_type += choice_type
        result_layer_dict = copy.deepcopy(layer_dict)
        result_layer_dict["params"] = params
        return result_layer_dict, res_mutate_type

    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        self.count += 1
        child_model_name = list(layer_dict.keys())[0]
        new_name = child_model_name + "_" + str(self.count)
        child_model_layer_list = layer_dict[child_model_name]
        new_layer_list = [child_model_layer_list[0]]
        child_model_layer_list = child_model_layer_list[1:]
        for layer in child_model_layer_list:
            if random.choice(list(range(10))) > 3:
                new_layer_list.append(layer)
                continue
            new_layer, _ = self.mutate(layer)
            new_layer_list.append(new_layer)
        return {new_name: new_layer_list}, "child_model_mutate"

    def __get_value(self, para_constraint_dict: dict) -> (str, str):
        value = ""
        choice_type = ''
        dic = para_constraint_dict
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "torch.string":
                if "enum" in dic:
                    value = random.choice((dic["enum"]))
                    choice_type += str(value)
                    value = '"' + str(value) + '"'
                else:
                    value = dic["default"]
                    choice_type += 'default'
                    if value == "None":
                        pass
                    else:
                        value = '"' + value + '"'
            elif type == "torch.bool":
                value = random.choice([True, False])
                choice_type += str(value)
            elif type == "float":
                value = random.random().__str__()
                choice_type += 'legal float'
            elif type == "int":
                if "structure" in dic and "range" in dic:
                    structure = dic["structure"]
                    structure = random.choice(dic["structure"]) if isinstance(structure,
                                                                              list) else structure
                    drange = dic["range"]
                    min_v = int(drange[0])
                    max_v = int(drange[1])

                    if structure == "integer":
                        value = random.choice([random.randint(min_v, max_v), min_v, max_v])
                        if value == min_v:
                            choice_type += 'min int'
                        elif value == max_v:
                            choice_type += 'max int'
                        else:
                            choice_type += 'legal int'
                    elif structure == "tuple":
                        value = tuple(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                        choice_type += 'legal tuple'
                else:
                    if 'default' in dic:
                        value = dic["default"]
                        choice_type += 'default'
                    else:
                        value = 100
                        choice_type += 'what?'
        elif "default" in dic:
            value = dic["default"]
            choice_type += 'default'
        else:
            value = "1"
        return value, choice_type


class JittorMutator(Mutator):
    def mutate(self, layer_dict: dict) -> (dict, str):
        return layer_dict, "hei_hei"

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def get_value(self, para_constraint_dict: dict) -> str:
        pass


class TensorFlowMutator(Mutator):
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

    def __init__(self):
        super().__init__()
        self.count = 0

    def __get_library_name(self):
        return "tensorflow"

    def mutate(self, layer_dict: dict) -> (dict, str):
        if "layer" in layer_dict.keys():
            abstract_layer_name = layer_dict["layer"]
        else:
            abstract_layer_name = list(layer_dict.keys())[0]
        if abstract_layer_name == "cat":
            return layer_dict, "dont mutate this one"
        if not database.is_abstract_api_name_valid(abstract_layer_name):
            return self.child_model_mutate(layer_dict)
        if random.choice([1, 2]) == 1:
            print("api_name_mutate")
            return self.api_name_mutate(layer_dict)
        else:
            print("api_para_mutate")
            return self.api_para_mutate(layer_dict)

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        abstract_layer_name = layer_dict["layer"]
        implicit_layer_name = database.get_implicit_api_name("tensorflow", abstract_layer_name)
        end_with = implicit_layer_name[-2:]
        valid_similarity_dict = database.get_implicit_api_similarity_valid("tensorflow", implicit_layer_name)
        if len(valid_similarity_dict) == 0:
            return layer_dict, "no mutate"
        else:
            new_implicit_layer_name = roulette_wheel_selection(valid_similarity_dict)
            if end_with in ["1D", "2D", "3D"]:
                while end_with not in new_implicit_layer_name:
                    new_implicit_layer_name = roulette_wheel_selection(valid_similarity_dict)
            else:
                pass
            new_abstract_layer_name = database.get_abstract_api_name("tensorflow", new_implicit_layer_name)
            # return res_mutate_type as label
            res_mutate_type = new_abstract_layer_name
            abstract_layer_info = database.get_abstract_layer_info(new_abstract_layer_name)
            required_list = abstract_layer_info["inputs"]["required"]
            param_constraints = abstract_layer_info["constraints"]
            new_para_dict = {}
            _old_para_dict = layer_dict["params"]
            old_para_dict = self.__convert_to_tf(abstract_layer_name, _old_para_dict)
            for p in old_para_dict.items():
                if p[0] in param_constraints.keys():
                    new_para_dict[p[0]] = p[1]
            for param_name in required_list:
                if param_name not in new_para_dict.keys():
                    new_para_dict[param_name] = self.__get_value(param_constraints[param_name])[0]
            result_layer_dict = dict([("layer", new_abstract_layer_name),
                                      ("params", new_para_dict),
                                      ("in", layer_dict["in"]),
                                      ("out", layer_dict["out"]),
                                      ])
            return result_layer_dict, res_mutate_type

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        abstract_layer_name = layer_dict["layer"]
        now_api_para_data = database.get_abstract_layer_info(abstract_layer_name)["constraints"]
        _old_para_dict = copy.deepcopy(layer_dict["params"])
        old_para_dict = self.__convert_to_tf(abstract_layer_name, _old_para_dict)
        param_to_mutate = self.__random_param(now_api_para_data) if len(list(now_api_para_data.keys())) > 1 else \
        list(now_api_para_data.keys())[0]
        res_mutate_type = param_to_mutate
        value, choice_type = self.__get_value(now_api_para_data[param_to_mutate])
        new_para_dict = copy.deepcopy(old_para_dict)
        new_para_dict[param_to_mutate] = value
        # return res_mutate_type as label
        res_mutate_type += f":{choice_type}"
        result_layer_dict = dict([("layer", layer_dict["layer"]),
                                  ("params", new_para_dict),
                                  ("in", layer_dict["in"]),
                                  ("out", layer_dict["out"]),
                                  ])
        return result_layer_dict, res_mutate_type

    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        # TODO
        self.count += 1
        child_model_name = list(layer_dict.keys())[0]
        new_name = child_model_name + "_" + str(self.count)
        child_model_layer_list = layer_dict[child_model_name]
        new_layer_list = [child_model_layer_list[0]]
        child_model_layer_list = child_model_layer_list[1:]
        for layer in child_model_layer_list:
            if random.choice(list(range(10))) > 3:
                new_layer_list.append(layer)
                continue
            new_layer, _ = self.mutate(layer)
            new_layer_list.append(new_layer)
        return {new_name: new_layer_list}, "child_model_mutate"

    def __random_param(self, data: dict) -> str:
        """
        The parameters to be mutated are selected according to the roulette wheel method
        @param data: (dict)Parameters dictionary
        @return: param: (str)A parameter for this mutation
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

    def __convert_to_tf(self, abstract_layer_name: str, para_dict: dict) -> dict:
        """
        Convert an abstract parameter name to a corresponding parameter name
        @param abstract_layer_name: (str)Name of the abstraction layer
        @param para_dict: (dict)Abstract argument list
        @return: res_para_dict: (dict)Transformed argument list
        """
        res_para_dict = {}

        for param in para_dict:
            implicit_param_name = database.get_implicit_para_name(self.__get_library_name(), abstract_layer_name, param)
            # implicit_param_name = self.__get_implicit_para_name(abstract_layer_name, param)
            if implicit_param_name != "None":
                res_para_dict[implicit_param_name] = para_dict[param]
            else:
                pass
        # modify padding
        if "padding" in para_dict.keys():
            res_para_dict["padding"] = '"valid"' if para_dict["padding"] == 0 else '"same"'
        return res_para_dict

    def __get_value(self, para_constraint_dict: dict) -> (str, str):
        """
        Get the values for the selected parameters
        @param para_constraint_dict: (dict)A dictionary of specific values for the selected parameters, which is different from the input dictionary for random_param
        @return: (value: (str)A valid value for the parameter is selected
                  choice_type: (str) The label value corresponding)
        """
        value = ""
        choice_type = ""
        dic = para_constraint_dict
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "tf.string":
                if "enum" in dic:
                    value = random.choice((dic["enum"]))
                    choice_type += str(value)
                    value = "'" + str(value) + "'"
                else:
                    value = dic["default"]
                    choice_type += "default"
                    if value == "None":
                        pass
                    else:
                        value = "'" + str(value) + "'"
            elif type == "tf.bool":
                value = random.choice([True, False])
                choice_type += str(value)
            elif type == "float":
                value = random.random().__str__()
                choice_type += "legal float"
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
                        value = random.choice([random.randint(min_v, max_v), min_v, max_v])
                        if value == min_v:
                            choice_type += "min int"
                        elif value == max_v:
                            choice_type += "max int"
                        else:
                            choice_type += "legal int"
                    elif structure == "tuple":
                        value = tuple(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                        choice_type += "legal tuple"
                    elif structure == "list":
                        value = list(random.randint(min_v, max_v) for _ in range(dic["shape"]))
                        choice_type += "legal list"
                    elif structure == "tuple_of_tuples":
                        value = tuple(tuple(random.randint(min_v, max_v) for _ in range(2)) for _ in
                                      range(dic["shape"]))
                        choice_type += "legal tuple"
                else:
                    if "default" in dic:
                        value = dic["default"]
                        choice_type += "default"
                    else:
                        value = 100
                        choice_type += "what?"
            elif "default" in dic:
                value = dic["default"]
                choice_type += "default"
        else:
            value = "1"
        return value, choice_type


def get_mutator(library: str) -> Mutator | None:
    if library == "torch":
        return TorchMutator()
    elif library == "jittor":
        return JittorMutator()
    elif library == "tensorflow":
        return TensorFlowMutator()
    else:
        return None


# mutator = get_mutator("torch")
# lenetSeed = database.get_seed("LeNet")
# googlenetSeed = database.get_seed("googlenet")

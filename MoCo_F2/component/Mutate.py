import copy
import traceback
from abc import ABC, abstractmethod
import random
from _database import database


def roulette_wheel_selection(dictionary):
    total_value = sum(dictionary.values())
    random_number = random.uniform(0, total_value)
    current_sum = 0
    for key, value in dictionary.items():
        current_sum += value
        if current_sum >= random_number:
            return key


class Barrel:
    def __init__(self):
        # 桶
        self.b: dict[str: int] = {}
        return

    def add_elements(self, ele_list: list[str]):
        for ele in ele_list:
            self.add_new_element(ele)
        return

    def add_new_element(self, ele_name: str):
        if ele_name not in self.b.keys():
            self.b[ele_name] = 0
        return

    def add_one(self, ele_name: str):
        if ele_name in self.b.keys():
            self.b[ele_name] += 1
        return

    def calculate_unbalance_point(self, ele_range: list[str] | None = None) -> int:
        if ele_range is None:
            point_list = list(self.b.values())
        else:
            point_list = []
            for ele in ele_range:
                if ele in self.b.keys():
                    point_list.append(self.b[ele])
        return max(point_list) - min(point_list)

    def find_prefer(self, ele_range: list[str] | None = None):
        current_min = 10000000
        current_prefer = ""
        if ele_range is None:
            ele_range = list(self.b.keys())
        if len(ele_range) == 0:
            return "no prefer"
        elif len(ele_range) == 1:
            return ele_range[0]
        else:
            for ele in ele_range:
                if ele in self.b.keys():
                    if self.b[ele] < current_min:
                        current_min = self.b[ele]
                        current_prefer = ele
        return current_prefer

    def show_barrel_report(self) -> str:
        report = ""
        for ele in self.b.keys():
            report += f"{ele} : {str(self.b[ele])} \n"
        return report


barrel_point = 10


class Mutator(ABC):
    def __init__(self):
        return

    @abstractmethod
    def mutate(self, layer_dict: dict) -> (dict, str):
        # 随机选择api替换和参数替换，然后返回变异后的新layer_dict以及变异信息（用于剪枝）
        # 注意，此处传入的都是针对抽象层面的layer_dict
        pass

    @abstractmethod
    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    @abstractmethod
    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    @abstractmethod
    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        # 当进行子模型变异时，传入的不是一层而是一个子模型了
        pass


class TorchMutator(Mutator):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.barrel_report_count = 0
        all_api_list = database._Database__all_api_list["abstract"]
        self.api_barrel = Barrel()
        self.api_barrel.add_elements(all_api_list)
        self.para_barrel = {}
        for api in all_api_list:
            self.para_barrel[api] = Barrel()
            paras = list(database.get_abstract_layer_info(api)["constraints"].keys())
            self.para_barrel[api].add_elements(paras)

    def barrel_report(self):
        pth = f"../report/barrel_report{str(self.barrel_report_count)}.txt"
        self.barrel_report_count += 1
        f = open(pth, "w", encoding="utf-8")
        f.close()
        f = open(pth, "a", encoding="utf-8")
        f.write(self.api_barrel.show_barrel_report())
        f.write("\n\n=================\n\n")
        for key in self.para_barrel.keys():
            f.write(f"{key}:\n")
            f.write(f"{self.para_barrel[key].show_barrel_report()}\n")
            f.write("\n\n=================\n\n")
        f.close()
        return

    def mutate(self, layer_dict: dict) -> (dict, str):
        if "layer" in layer_dict.keys():
            abstract_layer_name = layer_dict["layer"]
        else:
            abstract_layer_name = list(layer_dict.keys())[0]
        if abstract_layer_name == "cat" or abstract_layer_name == "add":
            return layer_dict, "dont mutate this one"
        if not database.is_abstract_api_name_valid(abstract_layer_name):
            return self.child_model_mutate(layer_dict)
        if random.choice([1, 2, 3]) == 1:
            ld, _ = self.api_name_mutate(layer_dict)
        else:
            ld = layer_dict
        if random.choice([1, 2]) == 1:
            return self.api_name_mutate(ld)
        else:
            return self.api_para_mutate(ld)

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        try:
            abstract_layer_name = layer_dict["layer"]
            implicit_layer_name = database.get_implicit_api_name("torch", abstract_layer_name)
            valid_similarity_dict = database.get_implicit_api_similarity_valid("torch", implicit_layer_name)
            if len(valid_similarity_dict) == 0:
                return layer_dict, "no mutate"
            else:
                new_implicit_layer_name = roulette_wheel_selection(valid_similarity_dict)
                new_abstract_layer_name = database.get_abstract_api_name("torch", new_implicit_layer_name)

                # 桶处理
                # range_implicit = list(valid_similarity_dict.keys())
                # range_abstract = []
                # for implicit_layer_name_ in range_implicit:
                #     range_abstract.append(database.get_abstract_api_name("torch", implicit_layer_name_))
                # if self.api_barrel.calculate_unbalance_point(range_abstract) > barrel_point:
                #     new_abstract_layer_name = self.api_barrel.find_prefer(range_abstract)
                self.api_barrel.add_one(new_abstract_layer_name)
                # 桶处理

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
        except Exception:
            f = open("../report/error_report.txt", "a", encoding="utf-8")
            f.write("this one error while name mutate: " + str(layer_dict) + "\n")
            f.write(traceback.format_exc())
            f.write("\n================\n")
            f.close()
            return layer_dict, "no mutate"

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        try:
            now_api_para_data = database.get_abstract_layer_info(layer_dict["layer"])["constraints"]
            required = database.get_abstract_layer_info(layer_dict["layer"])["inputs"]["required"]
            optional = database.get_abstract_layer_info(layer_dict["layer"])["inputs"]["optional"]
            params = copy.deepcopy(layer_dict["params"])
            choice_list = required
            for _ in optional:
                choice_list.append(_)
            no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                              'num_features']
            for no_mutate_para in no_mutate_pool:
                if no_mutate_para in choice_list:
                    choice_list.remove(no_mutate_para)
            if len(choice_list) == 0:
                return layer_dict, 'no mutate'
            param_to_mutate = random.choice(choice_list)

            # 桶处理
            abstract_layer_name = layer_dict["layer"]
            barrel = self.para_barrel[abstract_layer_name]
            # if barrel.calculate_unbalance_point(choice_list) > barrel_point:
            #     param_to_mutate = barrel.find_prefer(choice_list)
            barrel.add_one(param_to_mutate)
            # 桶处理

            res_mutate_type = abstract_layer_name + "|" + str(param_to_mutate)
            value, choice_type = self.__get_value(now_api_para_data[param_to_mutate])
            params[param_to_mutate] = value
            res_mutate_type += "|" + choice_type
            result_layer_dict = copy.deepcopy(layer_dict)
            result_layer_dict["params"] = params
            return result_layer_dict, res_mutate_type
        except Exception:
            f = open("../report/error_report.txt", "a", encoding="utf-8")
            f.write("this one error while para mutate: " + str(layer_dict))
            f.write("\n================\n")
            f.close()
            return layer_dict, "no mutate"

    def child_model_mutate(self, layer_dict: dict) -> (dict, str):
        self.count += 1
        child_model_name = list(layer_dict.keys())[0]
        new_name = child_model_name + "_" + str(self.count)
        child_model_layer_list = layer_dict[child_model_name]
        new_layer_list = [child_model_layer_list[0]]
        child_model_layer_list = child_model_layer_list[1:]
        for layer in child_model_layer_list:
            if random.choice(list(range(10))) > 2:
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
                    if drange is not None:
                        min_v = int(drange[0])
                        max_v = int(drange[1])
                    else:
                        min_v = 1
                        max_v = 8

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
    def __init__(self):
        super().__init__()
        self.count = 0

    def mutate(self, layer_dict: dict) -> (dict, str):
        if "layer" in layer_dict.keys():
            abstract_layer_name = layer_dict["layer"]
        else:
            abstract_layer_name = list(layer_dict.keys())[0]
        if abstract_layer_name == "cat" or abstract_layer_name == "add":
            return layer_dict, "dont mutate this one"
        if not database.is_abstract_api_name_valid(abstract_layer_name):
            return self.child_model_mutate(layer_dict)
        if random.choice([1, 2]) == 1:
            return self.api_name_mutate(layer_dict)
        else:
            return self.api_para_mutate(layer_dict)

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        abstract_layer_name = layer_dict["layer"]
        implicit_layer_name = database.get_implicit_api_name("jittor", abstract_layer_name)
        valid_similarity_dict = database.get_implicit_api_similarity_valid("jittor", implicit_layer_name)
        if len(valid_similarity_dict) == 0:
            return layer_dict, "no mutate"
        else:
            new_implicit_layer_name = roulette_wheel_selection(valid_similarity_dict)
            new_abstract_layer_name = database.get_abstract_api_name("jittor", new_implicit_layer_name)
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
                    new_para_dict[param_name] = self.__get_value(param_constraints[param_name], param_name)[0]
            return {"layer": new_abstract_layer_name, "params": new_para_dict,
                    "in": layer_dict["in"], "out": layer_dict["out"]}, new_abstract_layer_name

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        now_api_para_data = database.get_abstract_layer_info(layer_dict["layer"])["constraints"]
        required = database.get_abstract_layer_info(layer_dict["layer"])["inputs"]["required"]
        optional = database.get_abstract_layer_info(layer_dict["layer"])["inputs"]["optional"]
        params = copy.deepcopy(layer_dict["params"])
        choice_list = required
        for _ in optional:
            choice_list.append(_)
        no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                          'num_features']
        for no_mutate_para in no_mutate_pool:
            if no_mutate_para in choice_list:
                choice_list.remove(no_mutate_para)
        if len(choice_list) == 0:
            return layer_dict, 'no mutate'
        param_to_mutate = random.choice(choice_list)
        res_mutate_type = str(param_to_mutate)
        value, choice_type = self.__get_value(now_api_para_data[param_to_mutate], param_to_mutate)
        params[param_to_mutate] = value
        res_mutate_type += choice_type
        result_layer_dict = copy.deepcopy(layer_dict)
        result_layer_dict["params"] = params
        return result_layer_dict, res_mutate_type
        pass

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
        pass

    def __get_value(self, dic: dict, para_name: str):  # now, get_value will return (value, choice_type)
        value = ""
        choice_type = ''
        if "dtype" in dic:
            dtype = dic["dtype"]
            type = random.choice(dtype) if isinstance(dtype, list) else dtype
            if type == "str":
                if "range" in dic:
                    value = random.choice((dic["range"]))
                    choice_type += str(value)
                    value = str(value)
                else:
                    value = dic["default"]
                    choice_type += 'default'
                    if value == "no default":
                        value = 'None'
                        choice_type += 'None'
                    else:
                        value = value
                        choice_type += 'default'
            elif type == "bool":
                value = random.choice([True, False])
                choice_type += str(value)
            elif type == "float":
                value = random.random().__str__()
                choice_type += 'legal float'
            elif type == "int":
                if para_name == 'dilation':
                    _min, _max = 1, 6
                elif para_name == 'padding':
                    _min, _max = 0, 8
                elif para_name == 'groups':
                    _min, _max = 1, 3
                elif para_name == 'kernel_size':
                    _min, _max = 1, 8
                elif para_name == 'stride':
                    _min, _max = 1, 4
                elif 'channels' in para_name or 'features' in para_name or 'size' in para_name:
                    _min, _max = 1, 512
                else:
                    _min, _max = 1, 8
                min_v = _min
                max_v = _max
                value = random.choice([random.randint(min_v, max_v), min_v, max_v])
                if value == min_v:
                    choice_type += 'min int'
                elif value == max_v:
                    choice_type += 'max int'
                else:
                    choice_type += 'legal int'
            elif type == "tuple":
                value = (random.randint(1, 8), random.randint(1, 8))
                choice_type = 'legal tuple'
            else:
                if dic['default'] != 'no default':
                    value = dic["default"]
                    choice_type += 'default'
                else:
                    value = 'None'
                    choice_type += 'None'
        else:
            if dic['default'] != 'no default':
                value = dic["default"]
                choice_type += 'default'
            else:
                value = 'None'
                choice_type += 'None'
        return value, choice_type


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
        if abstract_layer_name == "cat" or abstract_layer_name == "add":
            return layer_dict, "dont mutate this one"
        if not database.is_abstract_api_name_valid(abstract_layer_name):
            return self.child_model_mutate(layer_dict)
        if random.choice([1, 2]) == 1:
            # print("api_name_mutate")
            return self.api_name_mutate(layer_dict)
        else:
            # print("api_para_mutate")
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
                if layer["layer"] == "add" or layer["layer"] == "cat":
                    new_layer = copy.deepcopy(layer)
                else:
                    new_layer = {"in": layer["in"], "out": layer["out"], "layer": layer["layer"],
                                 "params": self.__convert_to_tf(layer["layer"], layer["params"])}
                new_layer_list.append(new_layer)
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


if __name__ == "__main__":
    mutator = get_mutator("torch")
    # net = "alexnet"
    # seed = database.get_seed(net)
    # for layer in seed[net]:
    #     for i in range(100):
    #         tmp = copy.deepcopy(layer)
    #         tmp = mutator.mutate(tmp)
    #         print(str(i) + ":" + str(tmp))

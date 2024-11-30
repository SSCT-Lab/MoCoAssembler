import copy
import itertools
import json
import os
import random
import uuid
import yaml

from DS.Block import Block
from Tools.ConstraintChecker import check_block

tf_infos_path = f"Data/tf_layer_infos/layer"
tf_similarity_path = f"Data/tf_layer_similarity"
threshold = 0.5

__DTYPE = ["int", "float", "string", "boolean"]
__STRUCTURE = ["scalar", "tuple", "list"]

# Mutate mode
MIN_MODE = "min"
NORMAL_MODE = "normal"
MAX_MODE = "max"
MUTATE_MODES = [NORMAL_MODE, MIN_MODE, NORMAL_MODE, MAX_MODE, NORMAL_MODE]
RNN_CELL = ["tf.keras.layers.GRU", "tf.keras.layers.LSTM", "tf.keras.layers.SimpleRNN"]
IGNORE_CELL = ["tf.keras.layers.RNN"]
NO_MUTATE_CELL = ["tf.keras.layers.concatenate", "tf.keras.layers.add"]


def get_tf_api_list():
    file_list = os.listdir(tf_infos_path)
    return [_.replace(".json", "") for _ in file_list]


def load_api_info(api_name):
    path = f"{tf_infos_path}/{api_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["params"]


def load_api_similarity(api_name):
    path = f"{tf_similarity_path}/{api_name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.full_load(f)
    return data


class Mutator:
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
        self.api_list = get_tf_api_list()
        self.api_info = {}
        self.api_similarity = {}
        self.filtered_api_similarity = {}

        for api in self.api_list:
            self.api_info[api] = load_api_info(api)
            _api_similarity = load_api_similarity(api)
            _api_list = list(_api_similarity.keys())
            for _api in _api_list:
                if _api not in self.api_list:
                    _api_similarity.pop(_api)
            self.api_similarity[api] = _api_similarity

        for api in self.api_list:
            dimension = ''
            if '1D' in api or '1d' in api:
                dimension = '1D'
            elif '2D' in api or '2d' in api:
                dimension = '2D'
            elif '3D' in api or '3d' in api:
                dimension = '3D'
            _filtered_similarity = {}
            for name, probability in self.api_similarity[api].items():
                if name == api:
                    continue
                if probability <= threshold:
                    continue
                if dimension:
                    if ('1D' in name.upper() or '2D' in name.upper() or '3D' in name.upper()) and dimension not in name.upper():
                        continue
                _filtered_similarity[name] = probability
            self.filtered_api_similarity[api] = _filtered_similarity

    def mutate(self, block: Block):
        if block.api_name in NO_MUTATE_CELL:
            return block, "no nutate"
        if block.is_child_model:
            return self._child_model_mutate(block)
        else:
            return self._normal_mutate(block)

    def _normal_mutate(self, block: Block):
        assert not block.is_child_model
        if block.api_name not in self.api_list:
            return block, "no nutate"
        new_block, mutate_info = random.choice([self._api_name_mutate, self._api_param_mutate])(block)
        while not new_block.check_block_unknown():
            new_block, mutate_info = random.choice([self._api_name_mutate, self._api_param_mutate])(block)
        while not check_block(new_block):
            new_block, mutate_info = random.choice([self._api_name_mutate, self._api_param_mutate])(block)

        return new_block, mutate_info

    def _api_name_mutate(self, block: Block):
        ori_api_name = block.api_name
        similarity = self.filtered_api_similarity[ori_api_name]
        if len(similarity) > 0:
            new_api_name = random.choice(list(similarity.keys()))
            while new_api_name in IGNORE_CELL:
                new_api_name = random.choice(list(similarity.keys()))
        else:
            return block, "no mutate"

        _param_list = list(self.api_info[new_api_name].keys())
        _required_param_list = []
        for param in _param_list:
            if "default" not in self.api_info[new_api_name][param].keys():
                _required_param_list.append(param)

        new_params = {}
        if new_api_name.startswith("tf.nn."):
            for _param in _required_param_list:
                if _param not in ["input", "features", "x", "logits"]:
                    new_params[_param], _ = self._random_value(new_api_name, _param)
            if "input" in _param_list:
                new_params["input"] = block.input_symbols[0]
            elif "features" in _param_list:
                new_params["features"] = block.input_symbols[0]
            elif "x" in _param_list:
                new_params["x"] = block.input_symbols[0]
            elif "logits" in _param_list:
                new_params["logits"] = block.input_symbols[0]
        else:
            for _param in block.params.keys():
                if _param in _param_list:
                    new_params[_param] = block.params[_param]

            for _param in _required_param_list:
                if _param in block.params.keys():
                    new_params[_param] = block.params[_param]
                else:
                    new_params[_param], _ = self._random_value(new_api_name, _param)

        new_block = Block(new_api_name, new_params, block.node_name, block.input_symbols, block.output_symbols)
        if new_api_name.startswith("tf.nn."):
            new_block.block_type = 2
        mutate_info = f"ApiNameMutate, {new_api_name}"
        return new_block, mutate_info

    def _api_param_mutate(self, block: Block):
        param_list = list(self.api_info[block.api_name].keys())
        if len(param_list) == 0:
            return block, "no mutate"

        try:
            choice_param = self._get_random_param(param_list)
        except:
            choice_param = random.choice(param_list)
        value, value_info = self._random_value(block.api_name, choice_param)
        block.params[choice_param] = value
        mutate_info = f"ApiParaMutate, {choice_param}, {value_info}"
        return block, mutate_info

    def _child_model_mutate(self, block: Block):
        if not block.is_child_model:
            return block, "no mutate"
        new_block = copy.deepcopy(block)
        child_model = block.child_model
        graph_size = len(child_model.graph)
        choice_layer_number = random.randint(0, graph_size - 2)
        choice_layer = child_model.graph[choice_layer_number]
        choice_layer_name = choice_layer.node_name + str(uuid.uuid4())
        new_layer, layer_mutate_info = random.choice([self._api_name_mutate, self._api_param_mutate])(choice_layer)
        new_block.child_model.graph[choice_layer_number] = new_layer
        mutate_info = f"ChildModelMutate, {choice_layer_name}, {layer_mutate_info}"
        return new_block, mutate_info

    def _get_random_param(self, params_list):
        rare_probability = 0.005
        rare_count = 0
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

    def _random_value(self, api_name, param):
        def __get_single_value(dtype, range, mode):
            _value = "UNKNOWN"
            if dtype == "int":
                if mode == MAX_MODE:
                    _value = range[1]
                elif mode == MIN_MODE:
                    _value = range[0]
                else:
                    _value = random.randint(range[0], range[1])
            elif dtype == "float":
                if mode == MAX_MODE:
                    _value = _range[1]
                elif mode == range:
                    _value = range[0]
                else:
                    _value = random.uniform(range[0], range[1])
            return _value

        def __get_value(dtype, range_, structure, shape):
            _value = "UNKNOWN"
            _mode = "UNKNOWN"
            if structure == "scalar" or shape == 1:
                _mode = random.choice(MUTATE_MODES)
                _value = __get_single_value(dtype, range_, _mode)
                _value_info = f"{dtype}, scalar, {_mode if dtype in ['int', 'float'] else _value}"
            elif _structure in ["tuple", "list"]:
                _value_list = []
                _mode_list = []
                for i in range(shape):
                    _mode = random.choice(MUTATE_MODES)
                    _mode_list.append(_mode)
                    _value_list.append(__get_single_value(dtype, range_, _mode))

                _value = _value_list if _structure == "list" else tuple(_value_list)
                _mode = "list" + str(_mode_list) if _structure == "list" else "tuple" + str(tuple(_mode_list))

            return _value, _mode

        _api_info = self.api_info[api_name][param]
        _dtype_list, _range, _structure_list, _shape_list = _api_info["dtype"], _api_info["range"], _api_info["structure"], _api_info["shape"]

        if "default" in _api_info.keys():
            _default_list = _api_info["default"]
            _default = _default_list[0]
        _value = "UNKNOWN"
        _mode = "UNKNOWN"

        _dtype = random.choice(_dtype_list)
        _structure = random.choice(_structure_list)
        _shape = random.choice(_shape_list)
        if _dtype in ["int", "float"]:
            if len(_range) == 0:
                _range = [1, 8] if _dtype == "int" else [0.0, 1.0]
            _value, _mode = __get_value(_dtype, _range, _structure, _shape)
        elif _dtype == "string":
            if len(_range) > 0:
                _mode = random.choice(_range)
                _value = "'" + _mode + "'"
            else:
                _mode = _default
                _value = _default if _default == "None" else "'" + _mode + "'"

        elif _dtype == "boolean":
            _value = _mode = random.choice([True, False])

        return _value, str(_mode)

    def boundary_mutate(self, block: Block):
        if block.is_child_model:
            return [], []
        res = []
        res_tag = []
        api_name = block.api_name
        if api_name not in self.api_list:
            return [], []
        param_set = list(self.api_info[api_name].keys())
        tags = ["MIN", "BELOW", "LEGAL", "MAX", "OVER", "LEGAL"]
        for param in param_set:
            _api_info = self.api_info[api_name][param]
            _dtype_list, _range, _structure_list, _shape_list = _api_info["dtype"], _api_info["range"], _api_info[
                "structure"], _api_info["shape"]

            for _dtype in _dtype_list:
                if len(_range) == 0:
                    _range = [1, 8] if _dtype == "int" else [0.0, 1.0]

                if _dtype == "int" or _dtype == "float":
                    MIN, MAX = _range
                    if _dtype == "int":
                        choices = [MIN, MIN - 1, MIN + 1, MAX, MAX + 1, MAX - 1]
                    else:
                        choices = [MIN, MIN - 0.0001, MIN + 0.0001, MAX, MAX + 0.0001, MAX - 0.0001]

                    for _structure, _shape in zip(_structure_list, _shape_list):
                        if _structure == "scalar":
                            for i in range(len(choices)):
                                new_block = copy.deepcopy(block)
                                new_block.params[param] = choices[i]
                                tag = tags[i]
                                res.append(new_block)
                                res_tag.append(tag)
                        elif _structure == "tuple" or _structure == "list":
                            isList = _structure == "list"
                            choices2 = [
                                tuple(product) if not isList else list(product) for product in
                                itertools.product(choices, repeat=_shape)
                            ]
                            tags2 = [
                                tuple(product) for product in itertools.product(tags, repeat=_shape)
                            ]
                            for i in range(len(choices2)):
                                new_block = copy.deepcopy(block)
                                new_block.params[param] = choices2[i]
                                tag = tags2[i]
                                res.append(new_block)
                                res_tag.append(tag)
        return res, res_tag

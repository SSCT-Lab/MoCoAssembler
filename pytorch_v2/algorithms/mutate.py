import copy
import random
from depart import Single_Model
import os
import file_paths
import yaml


def get_similarity_dict(api_name: str) -> dict:
    path = os.path.join(file_paths.LAYER_SIMILARITY_PATH, api_name + '.yaml')
    f = open(path, 'r', encoding='utf-8')
    d = yaml.full_load(f)
    f.close()
    api_list = []
    temp = os.listdir(file_paths.LAYER_INFO_PATH)
    for name in temp:
        api_list.append(name[:-5])
    if api_name in d.keys():
        del d[api_name]
    modified_d = {}
    for name in d.keys():
        if name in api_list:
            modified_d[name] = d[name]

    # dim control, 1d, 2d, 3d
    if '1d' in api_name:
        pool = ['2d', '3d']
    elif '2d' in api_name:
        pool = ['1d', '3d']
    elif '3d' in api_name:
        pool = ['1d', '2d']
    else:
        return modified_d
    pool = pool
    modified_modified_d = {}
    for name in modified_d.keys():
        if not (pool[0] in name or pool[1] in name):
            modified_modified_d[name] = modified_d[name]
        else:
            continue
    return modified_modified_d


def get_constraint_dict(api_name: str) -> dict:
    path = os.path.join(file_paths.LAYER_INFO_PATH, api_name + '.yaml')
    f = open(path, 'r', encoding='utf-8')
    d = yaml.full_load(f)
    f.close()
    return d


def get_params(line: str) -> dict:  # 注: 解析单句的参数列表, 例如torch.nn.Conv2d(...), 必须带上参数名
    ana = line.split('(', 1)[1][:-1]
    ana = ana.replace(' ', '')
    temp_str = ''
    params_dict = {}
    index = 0
    while index < len(ana):
        i = index
        while i < len(ana) and ana[i] != '=':
            temp_str = temp_str + ana[i]
            i = i + 1
        name = copy.deepcopy(temp_str)
        i = i + 1
        temp_str = ''
        while i < len(ana) and not (ana[i] == ',' and not (ana[i + 1].isdigit())):
            temp_str = temp_str + ana[i]
            i = i + 1
        value = copy.deepcopy(temp_str)
        temp_str = ''
        if i < len(ana):
            i = i + 1
        params_dict[name] = value
        index = i
    return params_dict


def get_function(line: str) -> str:  # 注: 解析单句的函数名, 如torch.nn.Conv2d(...)
    return line.split('(', 1)[0]


def generate_line(api_name: str, params_dict: dict) -> str:
    new_params: str = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    return api_name + new_params


def get_value(dic: dict):  # now, get_value will return (value, choice_type)
    value = ""
    choice_type = ''
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
    else:
        value = dic["default"]
        choice_type += 'default'
    return value, choice_type


def roulette_wheel_selection(prob_dict):
    sum_prob = sum(prob_dict.values())
    rand = random.uniform(0, 1)
    proportion_list = [prob / sum_prob for prob in prob_dict.values()]
    dist_list = [abs(proportion_list[i] - rand) for i in range(len(proportion_list))]
    min_dist_index = dist_list.index(min(dist_list))
    return list(prob_dict.keys())[min_dist_index]


class Mutator:
    def __init__(self):  # 1 list for all api names, 2 dict for similarity and constraint, key api_name to access
        self.th = 0.5
        self.api_list = []
        temp = os.listdir(file_paths.LAYER_INFO_PATH)
        for name in temp:
            self.api_list.append(name[:-5])
        del temp
        self.api_similarity = {}
        self.api_constraint = {}
        for api_name in self.api_list:
            self.api_similarity[api_name] = get_similarity_dict(api_name)
            self.api_constraint[api_name] = get_constraint_dict(api_name)
        return

    def api_mutate(self, api: str) -> (str, str):  # now, api mutate will return newline and its mutate type(explicit)
        # no ( in api, don't mutate
        if '(' not in api:
            return api, 'no mutate'

        # not in layers range, don't mutate
        if 'torch.nn' not in api:
            return api, 'no mutate'

        # int fix: if there is no param name, add it

        # no = in api, that means it's special layer, don't mutate
        check_list = api.split('(', 1)[1].split(')')[0].split(',')
        for element in check_list:
            if len(check_list) == 1 and check_list[0] == '':
                break
            if '=' not in element:
                return api, 'no mutate'

        flag = random.choice(['para mutate', 'name mutate'])
        if flag == 'para mutate':
            return self.api_para_mutate(api)
        elif flag == 'name mutate':
            return self.api_name_mutate(api)
        else:
            return api, 'no mutate'

    def api_para_mutate(self, api: str) -> (str, str):
        api_name = get_function(api)
        if api_name not in self.api_list:
            return api, 'no mutate'
        para_dict = get_params(api)
        now_api_para_data = self.api_constraint[api_name]['constraints']
        new_para_dict = copy.deepcopy(para_dict)

        # shape limit
        choice_list = list(now_api_para_data.keys())
        no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                          'num_features']
        for no_mutate_para in no_mutate_pool:
            if no_mutate_para in choice_list:
                choice_list.remove(no_mutate_para)
        if len(choice_list) == 0:
            return api, 'no mutate'
        param_to_mutate = random.choice(choice_list)

        res_mutate_type = ''
        res_mutate_type += str(param_to_mutate)
        value, choice_type = get_value(now_api_para_data[param_to_mutate])
        new_para_dict[param_to_mutate] = value
        new_line = generate_line(api_name, new_para_dict)
        return new_line, res_mutate_type + ': ' + choice_type

    def api_name_mutate(self, api: str) -> (str, str):
        api_name = get_function(api)
        if api_name not in self.api_list:
            return api, 'no mutate'
        para_dict = get_params(api)

        # choose by probability in sim dict
        now_api_sim_dict = self.api_similarity[api_name]
        new_api_name = roulette_wheel_selection(now_api_sim_dict)

        # threshold but not completely locked
        count = 0
        while now_api_sim_dict[new_api_name] < self.th and count < 10:
            new_api_name = roulette_wheel_selection(now_api_sim_dict)
            count = count + 1

        # param adaptation
        now_api_para_data = self.api_constraint[new_api_name]
        required_list = now_api_para_data['inputs']['required']
        params_constraint_dict = now_api_para_data['constraints']
        new_para_dict = {}
        for p in para_dict.items():
            if p[0] in params_constraint_dict.keys():
                new_para_dict[p[0]] = p[1]
        for param_name in required_list:
            if param_name not in new_para_dict.keys():
                new_para_dict[param_name] = get_value(params_constraint_dict[param_name])[0]

        new_api = generate_line(new_api_name, new_para_dict)
        return new_api, new_api_name

    def sequence_mutate(self, seq_api_list: list) -> list:
        result = []
        for api_str in seq_api_list:
            if random.randint(1, 10) < 3:
                result.append(self.api_mutate(api_str)[0])
            else:
                result.append(api_str)
        return result

    def child_model_mutate(self, model: Single_Model) -> Single_Model:
        result = Single_Model()
        result.execute = copy.deepcopy(model.execute)
        result.model_name = copy.deepcopy(model.model_name)
        result.init_sentence = copy.deepcopy(model.init_sentence)
        result.execute_sentence = copy.deepcopy(model.execute_sentence)
        result.other_parts = copy.deepcopy(model.other_parts)
        result.return_sentence = copy.deepcopy(model.return_sentence)
        result.block_visited = copy.deepcopy(model.block_visited)
        result.tag = 'block'
        names = model.declaration.keys()
        for name in names:
            if isinstance(model.declaration[name], str):
                if '(' not in model.declaration[name]:
                    dec = model.declaration[name]
                else:
                    dec = self.api_mutate(model.declaration[name])[0]
                result.declaration[name] = dec
            elif isinstance(model.declaration[name], list):
                dec = self.sequence_mutate(model.declaration[name])
                result.declaration[name] = dec
        return result


if __name__ == '__main__':
    m = Mutator()
    api1 = 'torch.nn.Sigmoid()'
    for i in range(100):
        print(m.api_mutate(api1))

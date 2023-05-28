import random
import re

import yaml
import file_paths
import tools
import Similarity as sim
import copy


class Mutator:
    # 维护了四个信息表(dict)：两种相似度表api_similarity_info_dict 和 layer_similarity_info_dict
    # 两种接口约束表api_constraint_dict 和 layer_constraint_dict
    # 维护了两个总览表(list)：api_list和layer_list，装所有接口名字
    def __init__(self):
        self.api_similarity_info_dict = {}
        self.layer_similarity_info_dict = {}
        self.s = sim.Similarity()
        self.api_constraint_dict = copy.deepcopy(self.s.all_api_info)
        self.s.set_path(file_paths.LAYER_INFO_PATH)
        self.layer_constraint_dict = copy.deepcopy(self.s.all_api_info)
        del self.s
        with open(file_paths.API_SIMILARITY_FILE_PATH, 'r') as f:
            self.api_similarity_info_dict = yaml.full_load(f.read())
            f.close()
        with open(file_paths.LAYER_SIMILARITY_FILE_PATH, 'r') as f:
            self.layer_similarity_info_dict = yaml.full_load(f.read())
            f.close()
        self.api_list = self.api_similarity_info_dict.keys()
        self.layer_list = self.layer_similarity_info_dict.keys()

    def api_mutate(self, api_str: str) -> (str, str):
        # 5.25修改：返回变异结果的同时，返回一个变异类型
        r = random.randint(0, 1)

        if 'conv' in api_str.lower():
            result = self.api_name_mutate(api_str)
            result = self.api_para_adapt(result)
            return result, 'api name mutate'
        # TODO：conv变参数的时候，约束需要满足一下，先不变    5.28 : conv只进行name_mutate

        result = ''
        mutype = ''
        if r == 0:
            result = self.api_name_mutate(api_str)
            result = self.api_para_adapt(result)
            mutype = 'api name mutate'
        else:
            result = self.api_para_mutate(api_str)
            mutype = 'api para mutate'

        return result, mutype

    def api_name_mutate(self, api_str: str) -> str:
        api_name: str = api_str.split('(')[0]
        if api_name in self.api_list:
            now_dict = self.api_similarity_info_dict[api_name]
            choice = tools.roulette_wheel_selection(now_dict)
            return choice + '(' + api_str.split('(')[1]
        elif api_name in self.layer_list:
            now_dict = self.layer_similarity_info_dict[api_name]
            choice = tools.roulette_wheel_selection(now_dict)
            # 5.22修改：此处新增传播参数相等的规则
            while self.layer_constraint_dict[choice]['extra_para'] != self.layer_constraint_dict[api_name]['extra_para']:
                choice = tools.roulette_wheel_selection(now_dict)
            return choice + '(' + api_str.split('(')[1]
        else:
            return api_str

    def api_para_adapt(self, api_str: str) -> str: # 5.20修改：完善了名字编译后的参数适配规则，包括str从range中选择等
        # TODO: 不完善的规则
        # original_api_info: dict = self.get_api_info(original_api_name)
        new_api_name: str = api_str.split('(')[0]
        new_api_info: dict = self.get_api_info(new_api_name)
        if 'inputs' not in new_api_info.keys():
            return new_api_name + '()'
        if len(new_api_info['inputs']['required']) == 0:
            return new_api_name + '()'

        # original_para_name_list = self.get_para_name_list(original_api_info)
        new_para_name_list = self.get_para_name_list(new_api_info)
        original_para_num_list = api_str.split('(')[1].split(')')[0].replace(' ', '').split(',')
        new_para_num_list = []
        for i in range(len(new_para_name_list)):
            if new_para_name_list[i][1] == 'optional':
                break
            if i >= len(original_para_num_list):
                now_para_name = new_para_name_list[i][0]
                now_type: list = new_api_info['constraints'][now_para_name]['dtype']
                new_para_num = self.generate_para(now_type[0])
                if now_type == 'str':
                    if 'range' in new_api_info['constraints'][now_para_name].keys():
                        str_range_list = new_api_info['constraints'][now_para_name]['range']
                        new_para_num = tools.random_select_from_list(str_range_list)
                    else:
                        new_para_num = 'None'
                new_para_num_list.append(new_para_num)
            else:
                # pattern = r'^[+-]?\d*\.?\d+$'
                now_para_name = new_para_name_list[i][0]
                now_type = new_api_info['constraints'][now_para_name]['dtype']
                para_type = tools.get_string_type(original_para_num_list[i])
                if para_type in now_type:
                    new_para_num = original_para_num_list[i]
                else:
                    new_para_num = self.generate_para(now_type[0])
                    if now_type == 'str':
                        if 'range' in new_api_info['constraints'][now_para_name].keys():
                            str_range_list = new_api_info['constraints'][now_para_name]['range']
                            new_para_num = tools.random_select_from_list(str_range_list)
                        else:
                            new_para_num = 'None'
                new_para_num_list.append(new_para_num)
        result = ''
        result = result + new_api_name
        result = result + '('

        # 5.28 修改：将参数名也加上

        for i in range(len(new_para_num_list)):
            result = result + new_para_name_list[i][0] + ' = ' + new_para_num_list[i] + ','
        result = result[:-1]
        result = result + ')'

        return result



    def api_para_mutate(self, api_str) -> str:
        api_name = api_str.split('(')[0]
        api_info = self.get_api_info(api_name)
        constraints = api_info['constraints']
        required_para_names = api_info['inputs']['required']
        optional_para_names = api_info['inputs']['optional']
        result_list = []
        for name in required_para_names:
            part1 = name + ' = '
            if name in constraints.keys():
                if 'str' in constraints[name]['dtype']:
                    if 'range' in constraints[name].keys():
                        range = constraints[name]['range']
                        val = random.choice(range)
                    else:
                        val = 'None'
                else:
                    ty = constraints[name]['dtype'][0]
                    val = self.generate_para(ty)
                result_list.append(part1 + val)
            else:
                result_list.append(part1 + '1')
        for name in optional_para_names:
            part1 = name + ' = '
            if name in constraints.keys():
                if 'str' in constraints[name]['dtype']:
                    if 'range' in constraints[name].keys():
                        range = constraints[name]['range']
                        val = random.choice(range)
                    else:
                        val = 'None'
                else:
                    ty = constraints[name]['dtype'][0]
                    val = self.generate_para(ty)
                result_list.append(part1 + val)
            else:
                continue
        result = api_name + '('
        for res_para in result_list:
            result = result + res_para + ', '
        if result[-2] == ',':
            result = result[:-2]
        result = result + ')'
        return result



    # ===========================
    # toolbox:
    # ===========================
    def generate_para(self, type: str) -> str:
        if type == 'int':
            return str(random.randint(1,8))
        elif type == 'str':
            return ''
        elif 'floa' in type:
            return str(random.random())
        elif 'tup' in type:
            return '(1,1)'
        elif 'lis' in type:
            return '[]'
        elif 'dic' in type:
            return '{}'
        elif 'bool' in type:
            return random.choice(['True', 'False'])
        else:
            return '1'

    def get_para_name_list(self, api_info: dict) -> list:
        para_name_list = []
        for para_name in api_info['inputs']['required']:
            para_name_list.append([para_name, 'required'])
        for para_name in api_info['inputs']['optional']:
            para_name_list.append([para_name, 'optional'])
        return para_name_list

    def get_api_info(self, api_name: str) -> dict:
        if api_name in self.api_list:
            return self.api_constraint_dict[api_name]
        elif api_name in self.layer_list:
            return self.layer_constraint_dict[api_name]
        else:
            return {}


if __name__ == '__main__':
    m = Mutator()
    api1 = 'jittor.nn.Conv(3,16,3)'
    # original_api_name = 'jittor.nn.Conv'
    # api_str = m.api_name_mutate(api1)
    # r = m.api_para_adapt(api_str)

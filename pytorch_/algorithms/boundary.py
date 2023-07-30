import copy

from assemble_complex import Model
import file_paths
import mutate
import os
import sys
from importlib import import_module
import openpyxl


class Boundary_Runner:
    def __init__(self, net_name: str = 'testnet'):
        self.bnum = 0
        self.bcount = 0
        self.api_list = []
        self.net_name = net_name

        MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
                      'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                      'densenet', 'BiLSTM', 'LSTM', 'GRU', 'googlenet']
        if not os.path.exists(os.path.join(file_paths.MAIN_PATH, 'boundary_models')):
            os.makedirs(os.path.join(file_paths.MAIN_PATH, 'boundary_models'))
            temp_path = os.path.join(file_paths.MAIN_PATH, 'boundary_models')
            for model_name in MODEL_LIST:
                if model_name != 'LOOP':
                    temp_model_path = os.path.join(temp_path, model_name)
                    if not os.path.exists(temp_model_path):
                        os.makedirs(temp_model_path)
                        print("folder created: " + temp_model_path)
                else:
                    continue
        if not os.path.exists(os.path.join(file_paths.MAIN_PATH, 'boundary_logs')):
            os.makedirs(os.path.join(file_paths.MAIN_PATH, 'boundary_logs'))
            temp_path = os.path.join(file_paths.MAIN_PATH, 'boundary_logs')
            for model_name in MODEL_LIST:
                if model_name != 'LOOP':
                    temp_model_path = os.path.join(temp_path, model_name)
                    if not os.path.exists(temp_model_path):
                        os.makedirs(temp_model_path)
                        print("folder created: " + temp_model_path)
                else:
                    continue

        path = os.path.join(file_paths.MAIN_PATH, 'boundary_models', self.net_name)
        sys.path.append(path)
        self.workbook = openpyxl.Workbook()
        self.blog_file_path = os.path.join(file_paths.MAIN_PATH, 'boundary_logs', self.net_name + '.xlsx')
        self.sheet = self.workbook.active
        self.sheet.append(['file_name', 'code', 'expect result', 'true result'])
        temp = os.listdir(file_paths.LAYER_INFO_PATH)
        for name in temp:
            self.api_list.append(name[:-5])
        del temp
        self.api_constraint = {}
        for api_name in self.api_list:
            self.api_constraint[api_name] = mutate.get_constraint_dict(api_name)
        return

    def sav(self):
        self.workbook.save(os.path.join(file_paths.MAIN_PATH, 'boundary_logs', self.net_name,
                                        self.net_name + '-' + str(self.bcount) + '.xlsx'))
        del self.workbook
        self.workbook = openpyxl.Workbook()
        self.sheet = self.workbook.active
        self.sheet.append(['file_name', 'code', 'expect result', 'true result'])

    def boundary_test(self, model: Model) -> int:
        if model.generation == 0:
            return 0
        # find the target dec:
        sentence: str = model.model.main_model.execute[-1]
        if 'self.' not in sentence:
            return 0
        name = sentence.split('self.', 1)[1].split('(', 1)[0]
        if name not in model.model.main_model.declaration.keys():
            return 0
        dec = model.model.main_model.declaration[name]
        if not isinstance(dec, str):
            return 0
        new_dec_list = self.generate_all_boundary_dec(dec)
        if len(new_dec_list) == 0:
            return 0
        temp_model: Model = copy.deepcopy(model)
        for dec_and_exp in new_dec_list:
            temp_model.model.main_model.declaration[name] = dec_and_exp[0]
            exp = dec_and_exp[1]
            if exp == 0:
                exp = 'FAIL'
            elif exp == 1:
                exp = 'SUCCESS'
            elif exp == 2:
                exp = 'DEPEND'
            file_name = temp_model.model.assemble_file(temp_model.generation,
                                                       temp_model.index,
                                                       self.bnum,
                                                       os.path.join(file_paths.MAIN_PATH,
                                                                    'boundary_models',
                                                                    model.model.net_name))
            self.bnum += 1
            result = self.test(file_name)
            true_result = 'SUCCESS' if result else 'FAIL'

            if exp == true_result:
                path = os.path.join(file_paths.MAIN_PATH, 'boundary_models', self.net_name, file_name)
                os.unlink(path)
                pass
            else:
                self.sheet.append([file_name, dec_and_exp[0], exp, 'SUCCESS' if result else 'FAIL'])
        self.bcount += 1
        self.sav()
        return 0

    def test(self, file_name):
        # path = os.path.join(file_paths.MAIN_PATH, 'boundary_models', self.net_name, file_name)
        try:
            module_name = file_name.replace('.py', '')
            module = import_module(module_name)
            module.go()
            # os.unlink(path)
            return True
        except Exception:
            return False

    def generate_all_boundary_dec(self, dec: str) -> list[(str, int)]:  # dec, expect: 0:F 1:T 2:DEPEND
        function = mutate.get_function(dec)
        para_dict = mutate.get_params(dec)
        constraints = self.api_constraint[function]['constraints']
        para_list = list(para_dict.keys())
        no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                          'num_features']
        new_para_list = []
        for para in para_list:
            if para not in no_mutate_pool:
                new_para_list.append(para)
        para_list = new_para_list
        boundary_dec_list = []
        if len(para_list) == 0:
            return boundary_dec_list
        c = 0
        for para in para_list:
            if c >= 2:
                continue
            else:
                c += 1
            if 'dtype' not in constraints[para].keys():
                continue
            elif 'int' not in constraints[para]['dtype']:
                continue
            elif 'structure' not in constraints[para].keys():
                continue
            elif 'range' not in constraints[para].keys() or len(constraints[para]['range']) != 2:
                continue
            else:
                min_value = constraints[para]['range'][0]
                max_value = constraints[para]['range'][1]
                int_pool = [(min_value, 1), (min_value - 1, 0), (min_value + 1, 1),
                            (max_value, 1), (max_value - 1, 1), (max_value + 1, 2)]
                new_para_dict = copy.deepcopy(para_dict)
                if 'integer' in constraints[para]['structure']:
                    for value in int_pool:
                        new_para_dict[para] = value[0]
                        boundary_dec_list.append((mutate.generate_line(function, new_para_dict), value[1]))
                if 'tuple' in constraints[para]['structure']:
                    tuple_pool = []
                    for i in int_pool:
                        for j in int_pool:
                            t = (i[0], j[0])
                            if i[1] == 0 or j[1] == 0:
                                exp = 0
                            elif i[1] == 2 or j[1] == 2:
                                exp = 2
                            else:
                                exp = 1
                            tuple_pool.append((t, exp))
                    for value in tuple_pool:
                        new_para_dict[para] = value[0]
                        boundary_dec_list.append((mutate.generate_line(function, new_para_dict), value[1]))
        return boundary_dec_list


if __name__ == '__main__':
    _dec = "torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)"
    b = Boundary_Runner('testnet')

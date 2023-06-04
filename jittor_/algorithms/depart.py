import copy
import os.path

import file_paths


class Single_Model:
    def __init__(self):
        self.declaration: dict = {}  # layer <name> in execute->true <layer declaration>(str) / <seq declaration>(list)
        self.execute: list = []
        self.model_name: str = ''
        self.init_sentence: str = ''
        self.execute_sentence: str = ''
        self.other_parts: str = ''
        self.return_sentence: str = ''
        self.block_visited: int = 0
        self.tag: str = ''

    def assemble(self) -> str:  # assemble a single model to a str
        result = ''
        # class...
        result = result + 'class ' + self.model_name + '(nn.Module):\n'
        # init...
        result = result + self.init_sentence
        # declaration...
        keys = self.declaration.keys()
        for key in keys:
            dec = self.declaration[key]
            assert type(dec) in [str, list], 'type in dict error!!'
            if isinstance(dec, str):
                result = result + '        self.' + key + ' = ' + dec + '\n'
            else:
                dec_str = ''
                for layer in dec:
                    dec_str = dec_str + layer + ','
                dec_str = dec_str[:-1]
                result = result + '        self.' + key + ' = nn.Sequential(' + dec_str + ')\n'
        # other parts...
        result = result + self.other_parts
        # execute...
        result = result + self.execute_sentence
        result = result + connect_str_list(self.execute)
        # return...
        result = result + self.return_sentence

        return result


class Departed_Model:
    def __init__(self, model_name: str):
        self.net_name: str = model_name
        self.begin: str = ''
        self.main_model: Single_Model = None  # the type is Single Model
        self.block_dict: dict = {}  # child block dict, key is block name after 'class', value is <Single_Model>
        self.end: str = ''
        if model_name in ['ResNet18', 'ResNet50', 'InceptionV3', 'testnet']:
            self.get_model_from_file(model_name)

    def assemble_file(self, generation: int = 0, index: int = 1) -> str:
        path = file_paths.MUTATED_MODEL_PATH
        f = open(os.path.join(path, self.net_name + '-' + str(generation) + '-' + str(index) + '.py'),
                 'w', encoding='utf-8')
        f.write(self.begin)
        f.close()
        f = open(os.path.join(path, self.net_name + '-' + str(generation) + '-' + str(index) + '.py'),
                 'a', encoding='utf-8')
        f.write(self.main_model.assemble())
        for model in self.block_dict.values():
            f.write(model.assemble())
        f.write(self.end)
        f.close()
        print(self.net_name + '-' + str(generation) + '-' + str(index) + '.py' + '      successfully assembled')
        return self.net_name + '-' + str(generation) + '-' + str(index) + '.py'

    def get_model_from_file(self, model_name: str) -> None:
        temp_list = []
        with open(os.path.join(file_paths.MODEL_PATH, model_name + '.py'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        main_flag = True
        for index, line in enumerate(lines):  # Go to main model analysis
            if 'nn.Module' in line:
                begin_index = index
                end_index = index + 1
                while 'nn.Module' not in lines[end_index] and '__name__' not in lines[end_index]:
                    end_index = end_index + 1
                class_code_block = lines[begin_index:end_index]
                model: Single_Model = self.single_model_analyse(class_code_block)
                if main_flag:
                    main_flag = False
                    # self.net_name = model.model_name
                    # model.tag = 'main'
                    self.main_model = model
                    begin_list = copy.deepcopy(temp_list)
                    temp_list.clear()
                    self.begin = connect_str_list(begin_list)
                else:
                    # model.tag = 'block'
                    self.block_dict[model.model_name] = model
            elif '__name__' in line:
                temp_list.clear()
                temp_list.append(line)
            else:
                temp_list.append(line)
        self.end = connect_str_list(temp_list)
        for model in self.block_dict.values():
            model.tag = 'block'
        self.main_model.tag = 'main'
        return

    def single_model_analyse(self, line_list) -> Single_Model:
        # input: a line list starts with 'class' and ends with 'return x', as a single model
        # output: a standard single model
        # format: this input line list should be of special format
        lines = copy.deepcopy(line_list)
        n = len(lines)
        result = Single_Model()
        i = 0
        while i < n:
            if i == 0:
                result.model_name = lines[i].split('(')[0][6:]
                i = i + 1
                continue
            if 'def' in lines[i]:
                if 'init' in lines[i]:
                    index = i
                    result.init_sentence = ''.join([lines[index], lines[index + 1]])
                    index = index + 2
                    while 'def' not in lines[index]:
                        if 'self.' in lines[index] and 'Seq' not in lines[index]:  # analyse this line in dict
                            if 'self.modules()' in lines[index]:
                                index = index + 1
                                continue
                            line = lines[index].replace(' ', '').replace('\n', '')
                            name = line.split('=', 1)[0][5:]
                            declare = line.split('=', 1)[1]
                            result.declaration[name] = declare
                            index = index + 1
                        elif 'self' in lines[index] and 'Seq' in lines[index]:
                            line = lines[index].replace(' ', '').replace('\n', '')
                            name = line.split('=', 1)[0][5:]
                            index = index + 1
                            seq_list = []
                            while 'jittor.nn.' in lines[index] or 'ConvBNReLU' in lines[index] or 'InceptionV3Module' in lines[index] or 'SeparableConv2d' in lines[index]:
                                seq_element = lines[index].replace(' ', '').replace('\n', '')
                                if seq_element.endswith(','):
                                    seq_element = seq_element[:-1]
                                seq_list.append(seq_element)
                                index = index + 1
                            index = index + 1
                            result.declaration[name] = seq_list
                        else:
                            index = index + 1
                    i = index
                elif 'execute' in lines[i]:
                    index = i
                    result.execute_sentence = copy.deepcopy(lines[index])
                    index = index + 1
                    while 'return' not in lines[index]:
                        result.execute.append(copy.deepcopy(lines[index]))
                        index = index + 1
                    result.return_sentence = lines[index]
                    index = index + 1
                    i = index
                else:
                    index = i
                    temp_list = []
                    while 'return' not in lines[index]:
                        temp_list.append(lines[index])
                        index = index + 1
                    temp_list.append(lines[index])
                    index = index + 1
                    i = index
                    result.other_parts = connect_str_list(temp_list)
            else:
                i = i + 1
        return result


# ============================================
# toolbox
# ============================================


def connect_str_list(str_list: list) -> str:  # connect sentences in a list with ''
    return ''.join(str_list)


if __name__ == '__main__':
    dm = Departed_Model('xception')
    # dm.assemble_file()

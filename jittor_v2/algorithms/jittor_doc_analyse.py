import copy
import os

# import jittor.nn
import yaml
import tools
import file_paths


class Analyse:
    # analyse jittor.nn.py and get all api information in yaml
    # attributes: line_list: 一个列表，里面是nn文件里所有行.  layers_to_mutate_list: 一个列表，里面是在变异范围内的层名.
    # current_flag: 一个游标，用于遍历
    def __init__(self):
        self.nn_path = os.path.join(file_paths.NN_SOURCE_FILE_PATH, 'pool.py')
        self.f = open(self.nn_path, 'r')
        self.line_list = self.f.readlines()
        self.layers_to_mutate_list = []
        with open(os.path.join(file_paths.API_SIMILARITY_PATH, 'layers_to_mutate_new.yaml'), 'r') as f:
            content = yaml.full_load(f.read())
            self.layers_to_mutate_list = list(content.keys())
            self.layers_to_mutate_list.append('jittor.nn.relu')
            self.layers_to_mutate_list.append('jittor.nn.leaky_relu')
            self.layers_to_mutate_list.append('jittor.nn.relu6')
            self.layers_to_mutate_list.append('jittor.nn.gelu')
            self.layers_to_mutate_list.append('jittor.nn.softmax')
            f.close()
        self.current_flag = 0
        self.total_line = len(self.line_list)

    def get_line(self, index: int) -> str:
        return self.line_list[index]

    def get_annotation_block(self, block_start_line_index: int) -> str:
        # 传入class或def的起始行，
        index = block_start_line_index
        if '\'\'\'' not in self.line_list[index + 1]:
            return 'no description'
        block_str = self.line_list[index + 1]
        index = index + 2
        while self.line_list[index].find("\'\'\'") == -1:
            block_str = block_str + self.line_list[index]
            index = index + 1
        block_str = block_str.split('Exa')[0]
        block_str = block_str + '\n\'\'\''
        return block_str

    def get_api_declaration(self, start_line_index: int) -> str:
        index = start_line_index
        if ')' in self.line_list[index]:
            return self.line_list[start_line_index].split('def ')[1].split(')')[0] + ')'
        else:
            result = ''
            while ')' not in self.line_list[index]:
                result = result + self.line_list[index]
                index = index + 1
            result = result + self.line_list[index]
            result = result.replace('\n', '')
            return result.split('def ')[1].split(')')[0] + ')'

    def get_execute_declaration(self, start_line_index: int) -> str:
        index = start_line_index
        result = ''
        while 'execute' not in self.line_list[index]:
            index = index + 1
        result = result + self.line_list[index]
        result = result.replace('\n', '')
        return result.split('def ')[1].split(')')[0] + ')'

    def get_layer_declaration(self, start_line_index: int) -> str:
        layer_name = self.line_list[start_line_index].split(' ')[1].split('(')[0]
        index = start_line_index + 1
        while not self.line_list[index].startswith('    def __init__'):
            index = index + 1
        layer_para = self.line_list[index].split('__init__')[1]
        return (layer_name + layer_para).split(')')[0] + ')'

    def analyse_layer_in_this_line(self, start_line_index) -> dict:
        result_dictionary = {}
        annotation_block = self.get_annotation_block(start_line_index)
        layer_declaration = self.get_layer_declaration(start_line_index)
        execute_declaration = self.get_execute_declaration(start_line_index)
        lis = execute_declaration.replace(' ','').split('(')[1].split(')')[0].split(',')[1:]
        extra_para_num = 0
        for s in lis:
            if '=' not in s:
                extra_para_num = extra_para_num + 1
        extra_para_num = extra_para_num - 1


        # 5.22修改，新增一个extra_para


        # analyse 'api':
        result_dictionary['api'] = 'jittor.nn.' + layer_declaration

        # analyse 'constraints':
        layer_declaration_copy = copy.deepcopy(layer_declaration)
        layer_declaration_copy = layer_declaration_copy.replace(' ', '')
        layer_declaration_copy = layer_declaration_copy.split('(')[1].replace(')', '')
        para_list = layer_declaration_copy.split(',')[1:]
        annotation_block_copy = copy.deepcopy(annotation_block)
        info_list = annotation_block_copy.split(':')
        constraint_dictionary = {}
        for para in para_list:
            now_para_dictionary = {}
            now_para_dictionary['default'] = 'no default'
            now_para_dictionary['dtype'] = []
            # find 'default':
            if '=' in para:  # 5.19改动：完善自动识别默认类型
                default = para.split('=')[1]
                now_para_dictionary['default'] = default
                dtype = tools.get_string_type(default)
                now_para_dictionary['dtype'].append(dtype)

                para_name = para.split('=')[0].split(':')[0]
            else:
                para_name = para.split(':')[0]
            # find 'type':
            for i, info in enumerate(info_list):
                if info.startswith('type ' + para_name):
                    index = i + 1
                    type_info = info_list[index]
                    type_info = type_info.split(',')[0]
                    type_info = type_info.replace('\n', '')
                    type_info = type_info.replace(' ', '')
                    dtype_list = type_info.split('or')

                    for dtype in dtype_list:
                        if dtype not in now_para_dictionary['dtype']:
                            now_para_dictionary['dtype'].append(dtype)

                    # now_para_dictionary['dtype'] = dtype_list   modified...
            if len(now_para_dictionary['dtype']) == 0:
                now_para_dictionary['dtype'].append('int')
            constraint_dictionary[para_name] = now_para_dictionary
        result_dictionary['constraints'] = constraint_dictionary

        # analyse 'descp':
        result_dictionary['descp'] = annotation_block

        # analyse 'inputs':
        inputs_dictionary = {}
        optional_list = []
        required_list = []
        for para in para_list:
            if '=' in para:
                para_name = para.split('=')[0].split(':')[0]
                optional_list.append(para_name)
            else:
                para_name = para.split(':')[0]
                required_list.append(para_name)
        inputs_dictionary['optional'] = optional_list
        inputs_dictionary['required'] = required_list
        result_dictionary['inputs'] = inputs_dictionary
        result_dictionary['extra_para'] = extra_para_num
        result_dictionary['kind'] = 'layer'
        print(result_dictionary['api'].split('(')[0] + ' successfully analysed.')

        return result_dictionary

    def analyse_api_in_this_line(self, start_line_index) -> dict:
        result_dictionary = {}
        annotation_block = self.get_annotation_block(start_line_index)
        api_declaration = self.get_api_declaration(start_line_index)

        # analyse 'api':
        result_dictionary['api'] = 'jittor.nn.' + api_declaration.replace('     ', ' ').replace('    ', ' ')

        # analyse 'constraints':
        layer_declaration_copy = copy.deepcopy(api_declaration)
        layer_declaration_copy = layer_declaration_copy.replace(' ', '')
        layer_declaration_copy = layer_declaration_copy.split('(')[1].replace(')', '')
        para_list = layer_declaration_copy.split(',')[1:]
        annotation_block_copy = copy.deepcopy(annotation_block)
        info_list = annotation_block_copy.split(':')
        constraint_dictionary = {}
        for para in para_list:
            now_para_dictionary = {}
            now_para_dictionary['default'] = 'no default'
            now_para_dictionary['dtype'] = []
            # find 'default':
            if '=' in para:  # 5.19改动：完善自动识别默认类型
                default = para.split('=')[1]
                now_para_dictionary['default'] = default
                dtype = tools.get_string_type(default)
                now_para_dictionary['dtype'].append(dtype)

                para_name = para.split('=')[0].split(':')[0]
            else:
                para_name = para.split(':')[0]
            # find 'type':
            for i, info in enumerate(info_list):
                if info.startswith('type ' + para_name):
                    index = i + 1
                    type_info = info_list[index]
                    type_info = type_info.split(',')[0]
                    type_info = type_info.replace('\n', '')
                    type_info = type_info.replace(' ', '')
                    dtype_list = type_info.split('or')

                    for dtype in dtype_list:
                        if dtype not in now_para_dictionary['dtype']:
                            now_para_dictionary['dtype'].append(dtype)

                    # now_para_dictionary['dtype'] = dtype_list   modified...
            if len(now_para_dictionary['dtype']) == 0:
                now_para_dictionary['dtype'].append('int')
            constraint_dictionary[para_name] = now_para_dictionary
        result_dictionary['constraints'] = constraint_dictionary

        # analyse 'descp':
        result_dictionary['descp'] = annotation_block


        # 5.22修改：增加一个传入额外参数数量的属性

        # analyse 'inputs':
        inputs_dictionary = {}
        optional_list = []
        required_list = []

        extra_para_num = 0

        for para in para_list:
            if '=' in para:
                para_name = para.split('=')[0].split(':')[0]
                optional_list.append(para_name)
            else:
                para_name = para.split(':')[0]
                required_list.append(para_name)
                extra_para_num = extra_para_num + 1
        inputs_dictionary['optional'] = optional_list
        inputs_dictionary['required'] = required_list
        result_dictionary['inputs'] = inputs_dictionary
        result_dictionary['extra_para'] = extra_para_num
        result_dictionary['kind'] = 'api'
        print(result_dictionary['api'].split('(')[0] + ' successfully analysed.')

        return result_dictionary

    def analyse(self):

        layer_list = self.layers_to_mutate_list
        api_path = file_paths.API_INFO_PATH
        layer_path = file_paths.LAYER_INFO_PATH
        for index in range(self.total_line):
            now_dict = {}
            if self.line_list[index].startswith('class '):
                now_dict = self.analyse_layer_in_this_line(index)
                api_name = now_dict['api'].split('(')[0]
                if api_name in layer_list:
                    layer_list.remove(api_name)
                    name = now_dict['api'].split('(')[0] + '.yaml'
                    with open(os.path.join(layer_path, name), 'w') as f:
                        yaml.dump(now_dict, f)
                        f.close()
            elif self.line_list[index].startswith('def '):
                now_dict = self.analyse_api_in_this_line(index)
                api_name = now_dict['api'].split('(')[0]
                if api_name in layer_list:
                    layer_list.remove(api_name)
                    name = now_dict['api'].split('(')[0] + '.yaml'
                    with open(os.path.join(api_path, name), 'w') as f:
                        yaml.dump(now_dict, f)
                        f.close()
            else:
                continue
        return layer_list


if __name__ == '__main__':
    a = Analyse()
    remain = a.analyse()
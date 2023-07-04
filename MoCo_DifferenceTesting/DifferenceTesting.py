import os
import copy
import subprocess
from datetime import datetime
import shutil
from alive_progress import alive_bar
import argparse

# vars
MAIN_PATH = os.path.join(os.getcwd())
SEED_MODELS_PATH = os.path.join(MAIN_PATH, 'seed_models')
RESULT_PATH = os.path.join(MAIN_PATH, 'results')
GENERATED_MODELS_PATH = os.path.join(RESULT_PATH, 'generated_models')
SAVED_MODELS_PATH = os.path.join(RESULT_PATH, 'saved_models')
MODEL_LIST = ['TestNet', 'LeNet', 'AlexNet', 'GoogleNet', 'InceptionV3', 'MobileNet', 'ResNet18', 'ResNet50',
              'SqueezeNet', 'VGG16', 'VGG19']


# depart *****************************************************************
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
    def __init__(self, model_name: str = 'LeNet', mode: str = 'PyTorch'):
        assert mode in ['PyTorch', 'Jittor']
        self.net_name: str = model_name
        self.mode = mode  # jittor mode or pytorch mode
        self.begin: str = ''
        self.main_model: Single_Model = Single_Model()  # the type is Single Model
        self.block_dict: dict = {}  # child block dict, key is block name after 'class', value is <Single_Model>
        self.end: str = ''
        if model_name in MODEL_LIST:
            self.get_model_from_file(model_name)
            if self.mode == 'Jittor':
                self.begin = self.begin.replace('torch', 'jittor')
                self.main_model.execute_sentence = self.main_model.execute_sentence.replace('forward', 'execute')
                for key in self.block_dict.keys():
                    self.block_dict[key].execute_sentence = self.block_dict[key].execute_sentence.replace('forward',
                                                                                                          'execute')
                self.end = self.end.replace('torch', 'jittor')

    def assemble_file(self, generation: int = 0, test_dim: int = 1) -> str:
        path = os.path.join(GENERATED_MODELS_PATH)
        f = open(os.path.join(path, self.net_name + '-' + self.mode + '-' + str(generation) + '-' + str(test_dim) +
                              '.py'),
                 'w', encoding='utf-8')
        f.write(self.begin)
        f.close()
        f = open(os.path.join(path, self.net_name + '-' + self.mode + '-' + str(generation) + '-' + str(test_dim) +
                              '.py'),
                 'a', encoding='utf-8')
        f.write(self.main_model.assemble())
        for model in self.block_dict.values():
            f.write(model.assemble())
        f.write(self.end.replace('tbd', str(test_dim)))
        f.close()
        return self.net_name + '-' + self.mode + '-' + str(generation) + '-' + str(test_dim) + '.py'

    def get_model_from_file(self, model_name: str) -> None:
        temp_list = []
        with open(os.path.join(SEED_MODELS_PATH, model_name + '.py'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        main_flag = True
        for index, line in enumerate(lines):  # Go to main model analysis
            if 'nn.Module' in line:
                begin_index = index
                end_index = index + 1
                while 'nn.Module' not in lines[end_index] and 'if __name__ == ' not in lines[end_index]:
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
            elif 'if __name__ == ' in line:
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
                            while 'nn.' in lines[index] or 'ConvBNReLU' in lines[index] or 'InceptionV3Module' in lines[index] or 'SeparableConv2d' in lines[index]:
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
                elif 'forward' in lines[i]:
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


def connect_str_list(str_list: list) -> str:  # connect sentences in a list with ''
    return ''.join(str_list)


def get_time_now() -> str:
    now = datetime.now()
    time_now = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
        now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)
    return time_now


# Delete Folder Content
def dfc():
    for filename in os.listdir(GENERATED_MODELS_PATH):
        file_path = os.path.join(GENERATED_MODELS_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"无法删除文件: {file_path}，错误信息: {e}")


# run
def run_2_parallel_models(model_name_in_Jittor: str, model_name_in_PyTorch: str):
    model_j = model_name_in_Jittor
    model_p = model_name_in_PyTorch
    # path_j = os.path.join(GENERATED_MODELS_PATH, model_name_in_Jittor)
    # path_p = os.path.join(GENERATED_MODELS_PATH, model_name_in_PyTorch)
    # run jittor file:
    try:
        result = subprocess.run(['python',
                                 os.path.join(GENERATED_MODELS_PATH, model_name_in_Jittor).replace('\\', '/')],
                                capture_output=True, text=True, check=True)
        output_j = result.stdout
        error_j = 'no error'
    except subprocess.CalledProcessError as e:
        output_j = e.stdout
        error_j = e.stderr
        if 'Traceback' in error_j:
            error_j = 'Traceback' + error_j.split('Traceback', 1)[1]
        else:
            pass

    # run pytorch file:
    try:
        result2 = subprocess.run(['python',
                                  os.path.join(GENERATED_MODELS_PATH, model_name_in_PyTorch).replace('\\', '/')],
                                 capture_output=True, text=True, check=True)
        output_p = result2.stdout
        error_p = 'no error'
    except subprocess.CalledProcessError as e:
        output_p = e.stdout
        error_p = e.stderr

    if (error_j != 'no error' and error_p == 'no error') or (error_j == 'no error' and error_p != 'no error'):
        # log writing
        log_path = os.path.join(RESULT_PATH, 'ed_log.txt')
        f = open(log_path, 'a', encoding='utf-8')
        f.write('**************************************************************************************************\n')
        sentence_ed = model_j + ' and ' + model_p + ' show different action in error detection, models/logs saved\n'
        print(sentence_ed, end='')
        f.write(get_time_now() + ':\n')
        f.write(sentence_ed)
        f.write('the error of Jittor file is: \n' + error_j + '\n\n')
        f.write('while the error of PyTorch file is: \n' + error_p + '\n\n')
        f.write('models saved in ' + get_time_now().replace(':', '').replace(' ', '').replace('-', '') + '\n')
        f.write('**************************************************************************************************'
                '\n\n\n')
        # model saving
        time = get_time_now().replace(':', '').replace(' ', '').replace('-', '')
        file_path = os.path.join(SAVED_MODELS_PATH, time)
        os.mkdir(file_path)
        source = GENERATED_MODELS_PATH
        shutil.copy(os.path.join(source, model_j), file_path)
        shutil.copy(os.path.join(source, model_p), file_path)

    elif error_p == 'no error' and error_j == 'no error':
        if output_p in output_j:
            pass
        else:
            log_path = os.path.join(RESULT_PATH, 'os_log.txt')
            f = open(log_path, 'a', encoding='utf-8')
            f.write(
                '**************************************************************************************************\n')
            sentence_os = model_j + ' and ' + model_p + ' show different output with same model' \
                                                        ' and same input, models/logs saved\n'
            print(sentence_os, end='')
            f.write(get_time_now() + ':\n')
            f.write(sentence_os)
            f.write('the output shape of Jittor file is: \n' + output_j + '\n\n')
            f.write('while the output shape of PyTorch file is: \n' + output_p + '\n\n')
            f.write(
                '**************************************************************************************************'
                '\n\n\n')

            time = get_time_now().replace(':', '').replace(' ', '').replace('-', '')
            file_path = os.path.join(SAVED_MODELS_PATH, time)
            os.mkdir(file_path)
            source = GENERATED_MODELS_PATH
            shutil.copy(os.path.join(source, model_j), file_path)
            shutil.copy(os.path.join(source, model_p), file_path)

    elif error_p != 'no error' and error_j != 'no error':
        if error_j == error_p:
            return output_j, output_p, error_j, error_p
        log_path = os.path.join(RESULT_PATH, 'es_log.txt')
        f = open(log_path, 'a', encoding='utf-8')
        f.write(
            '**************************************************************************************************\n')
        sentence_es = model_j + ' and ' + model_p + ' show different error message with same model' \
                                                    ' and same input, logs saved\n'
        # print(sentence_ed, end='')
        f.write(get_time_now() + ':\n')
        f.write(sentence_es)
        f.write('the error of Jittor file is: \n' + error_j + '\n\n')
        f.write('while the error of PyTorch file is: \n' + error_p + '\n\n')
        f.write(
            '**************************************************************************************************\n\n\n')
    else:
        pass
    return output_j, output_p, error_j, error_p


# difference testing
def test(model_name: str) -> None:
    assert model_name in MODEL_LIST, 'unsupported model name'
    dm_jittor = Departed_Model(model_name, mode='Jittor')
    dm_pytorch = Departed_Model(model_name, mode='PyTorch')
    dm_template_jittor, dm_template_pytorch = copy.deepcopy(dm_jittor), copy.deepcopy(dm_pytorch)
    dm_template_jittor.main_model.execute.clear()
    dm_template_pytorch.main_model.execute.clear()
    sentence_list_jittor = dm_jittor.main_model.execute
    sentence_list_pytorch = dm_pytorch.main_model.execute
    assert len(sentence_list_jittor) == len(sentence_list_pytorch), 'seed models got some problem'
    with alive_bar(5 * len(sentence_list_jittor), bar='smooth', title='Difference Testing (Template: ' +
                                                                      model_name + ')') as bar:
        for generation in range(1, len(sentence_list_jittor) + 1):
            dm_template_jittor.main_model.execute.append(sentence_list_jittor[generation - 1])
            dm_template_pytorch.main_model.execute.append(sentence_list_pytorch[generation - 1])
            for test_dim in [1, 2, 3, 4, 5]:
                jittor_file = dm_template_jittor.assemble_file(generation=generation, test_dim=test_dim)
                pytorch_file = dm_template_pytorch.assemble_file(generation=generation, test_dim=test_dim)
                run_2_parallel_models(jittor_file, pytorch_file)
                bar()
            dfc()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')
    args = parser.parse_args()
    MODEL = args.MODEL
    assert MODEL in MODEL_LIST, 'MODEL name is wrong, the range is ' + str(MODEL_LIST)
    test(MODEL)

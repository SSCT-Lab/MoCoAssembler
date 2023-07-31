import copy
import os
import queue
import shutil

import torch.cuda
import yaml

import mutate
from depart import Departed_Model, Single_Model
from complex_models import get_seed_model
import run
import random
from alive_progress import alive_bar
import file_paths


class Model:
    def __init__(self, dm: Departed_Model, g: int, i: int, info: str = 'default', c: int = 0):
        self.model: Departed_Model = dm
        self.generation: int = g
        self.index: int = i
        self.mutate_info = info
        self.param_count = c


class Assembler_Complex:
    def __init__(self, model_name):
        self.seed_model: Departed_Model = get_seed_model(model_name)
        self.mutator = mutate.Mutator()
        # self.shape_fixer = shape_fix.ShapeFixer()
        self.n = 2

        MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
                      'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                      'densenet', 'BiLSTM', 'LSTM', 'GRU', 'googlenet']
        # initialize:
        if not os.path.exists(file_paths.MUTATED_MODEL_PATH):
            os.makedirs(file_paths.MUTATED_MODEL_PATH)
            temp_path = file_paths.MUTATED_MODEL_PATH
            for model_name in MODEL_LIST:
                if model_name != 'LOOP':
                    temp_model_path = os.path.join(temp_path, model_name)
                    if not os.path.exists(temp_model_path):
                        os.makedirs(temp_model_path)
                        print("folder created: " + temp_model_path)
                else:
                    continue
        if not os.path.exists(file_paths.SAVED_MODEL_PATH):
            os.makedirs(file_paths.SAVED_MODEL_PATH)
            temp_path = file_paths.SAVED_MODEL_PATH
            for model_name in MODEL_LIST:
                if model_name != 'LOOP':
                    temp_model_path = os.path.join(temp_path, model_name)
                    if not os.path.exists(temp_model_path):
                        os.makedirs(temp_model_path)
                        print("folder created: " + temp_model_path)
                else:
                    continue
        if not os.path.exists(file_paths.LOG_PATH):
            os.makedirs(file_paths.LOG_PATH)
            temp_path = file_paths.LOG_PATH
            for model_name in MODEL_LIST:
                if model_name != 'LOOP':
                    temp_model_path = os.path.join(temp_path, model_name)
                    if not os.path.exists(temp_model_path):
                        os.makedirs(temp_model_path)
                        print("folder created: " + temp_model_path)
                else:
                    continue

    def set_n(self, val: int):
        self.n = val

    def dfc(self, folder_path=os.path.join(file_paths.MUTATED_MODEL_PATH, 'testnet')):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    # 删除文件
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 递归删除子文件夹及其内容
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除文件: {file_path}，错误信息: {e}")

    def sort_dict_by_value(self, input_dict: dict) -> dict:
        sorted_dict = dict(sorted(input_dict.items(), key=lambda x: x[1]))
        return sorted_dict

    def assemble_code_tree(self):
        import sys
        sys.path.append(os.path.join(file_paths.MUTATED_MODEL_PATH, self.seed_model.net_name))
        main_model = self.seed_model.main_model  # this is what to mutate in generation
        child_model_dict = copy.deepcopy(self.seed_model.block_dict)  # child model name <str> -> <Single_Model>
        block_list = child_model_dict.keys()
        execute_list = copy.deepcopy(main_model.execute)
        original_model = copy.deepcopy(self.seed_model)
        original_model.main_model.declaration.clear()
        original_model.main_model.execute.clear()
        original_model: Model = Model(original_model, 0, 1)
        for name in main_model.declaration.keys():  # 写入不可变定义
            dec = main_model.declaration[name]
            if isinstance(dec, list):
                continue
            elif isinstance(dec, str):
                if 'torch.nn' in dec:
                    continue
                else:
                    flag = True
                    for block in block_list:
                        if block in dec:
                            flag = False
                    if flag:
                        original_model.model.main_model.declaration[name] = dec
                    else:
                        continue

        model_queue = queue.Queue()
        model_queue.put(original_model)

        generation = 1
        last_layer_count = 1
        mutate_info_dict = {}
        filename_to_model_dict = {}

        for sentence in execute_list:
            if model_queue.empty():
                break

            # no need to change, add this in all models in queue
            if 'self.' not in sentence or sentence.startswith('        if') or sentence.startswith(
                    '            ') or sentence.startswith('        for'):
                new_queue = queue.Queue()
                while not model_queue.empty():
                    temp: Model = model_queue.get()
                    temp.model.main_model.execute.append(sentence)
                    new_queue.put(temp)
                model_queue = new_queue
                continue

            model_list: list[Model] = []
            next_layer_count = 0

            # test
            # with open(os.path.join('D:/PythonProjects/test_dict', 'gen' + str(generation-2) + '.yaml'), 'w') as f:
            #     yaml.dump(mutate_info_dict, f)
            # test

            mutate_info_dict.clear()
            filename_to_model_dict.clear()
            with alive_bar(last_layer_count, bar='smooth',
                           title=self.seed_model.net_name + '~generation' + str(generation - 1) + '~test run') as bar:
                for i in range(last_layer_count):
                    temp_model: Model = model_queue.get()
                    file_name = temp_model.model.assemble_file(generation=temp_model.generation, index=temp_model.index)
                    run_flag, param_flag = run.run_single_model(file_name, temp_model.model.net_name)
                    torch.cuda.empty_cache()
                    # something else also need to be returned
                    bar()
                    if run_flag:
                        temp_model.param_count = param_flag
                        mutate_info = temp_model.mutate_info
                        if mutate_info not in mutate_info_dict.keys():
                            mutate_info_dict[mutate_info] = {}
                        mutate_info_dict[temp_model.mutate_info][file_name] = param_flag
                        filename_to_model_dict[file_name] = copy.deepcopy(temp_model)
                    else:
                        pass

            # now, this model list is for all models that has passed 'run', next they will get cut and trained. and
            # we have file_name to param count dict mutate_info_dict to cut, and a filename_to_model_dict to get Models.
            file_list: list[str] = []
            for mutate_info in mutate_info_dict.keys():
                now_dict = mutate_info_dict[mutate_info]
                if len(now_dict.keys()) == 0:
                    continue
                elif len(now_dict.keys()) == 1:
                    f = list(now_dict.keys())[0]  # file name
                    model = filename_to_model_dict[f]
                    file_list.append(f)
                else:
                    n = len(now_dict.keys())
                    if n % 2 == 1:
                        n = int(n / 2) + 1
                    else:
                        n = int(n / 2)
                    now_dict = self.sort_dict_by_value(now_dict)
                    for i in range(n):
                        file_list.append(list(now_dict.keys())[i])

            # now, we have a file list that filled with file names of models that have been cut,
            # next, they will be trained and judged

            with alive_bar(len(file_list), bar='smooth',
                           title=self.seed_model.net_name + '~generation' + str(generation - 1) + '~test train') as bar:
                for file_name in file_list:
                    train_flag = run.train_single_model(file_name, self.seed_model.net_name)
                    torch.cuda.empty_cache()
                    bar()
                    if train_flag:
                        model_list.append(filename_to_model_dict[file_name])
                    else:
                        continue

            count = len(model_list)
            next_layer_count = count * self.n

            for i in range(count):
                now_origin: Model = model_list[i]
                for j in range(self.n):
                    index = (now_origin.index - 1) * self.n + j + 1
                    temp = copy.deepcopy(now_origin)
                    temp.generation = generation
                    temp.index = index
                    now_line = sentence.replace(' ', '').replace('\n', '')
                    name = now_line.split('.', 1)[1].split('(', 1)[0]

                    # this part is to mutate this sentence and add to this generation
                    if name not in main_model.declaration.keys():
                        continue
                    else:
                        dec: list or str = main_model.declaration[name]
                        if isinstance(dec, list):
                            new_dec = self.mutator.sequence_mutate(dec)
                            temp.mutate_info = 'sequence mutate'
                            temp.model.main_model.declaration[name] = new_dec
                            temp.model.main_model.execute.append(sentence)
                        else:
                            if 'torch.nn.' in dec:
                                new_dec, mut_info = self.mutator.api_mutate(dec)
                                temp.model.main_model.declaration[name] = new_dec
                                temp.mutate_info = mut_info
                                temp.model.main_model.execute.append(sentence)
                            else:
                                for block_name in block_list:
                                    if block_name in dec:
                                        temp.mutate_info = 'child model mutate'
                                        child_model = child_model_dict[block_name]
                                        new_child_model = self.mutator.child_model_mutate(child_model)
                                        child_model_dict[block_name].block_visited += 1

                                        new_child_model.block_visited = child_model_dict[block_name].block_visited
                                        new_model_name = new_child_model.model_name + '_' + str(
                                            new_child_model.block_visited)
                                        new_child_model.init_sentence = new_child_model.init_sentence.replace(
                                            new_child_model.model_name, new_model_name)
                                        new_child_model.model_name = new_model_name

                                        temp.model.block_dict[new_model_name] = copy.deepcopy(new_child_model)
                                        temp.model.main_model.declaration[name] = dec.replace(block_name,
                                                                                              new_model_name)
                                        temp.model.main_model.execute.append(
                                            sentence.replace(block_name, new_model_name))

                    # ==================================================================================================
                    model_queue.put(temp)

            self.dfc(os.path.join(file_paths.MUTATED_MODEL_PATH, self.seed_model.net_name))
            generation = generation + 1
            last_layer_count = next_layer_count

        model_list: list[Model] = []
        next_layer_count = 0

        # test
        # with open(os.path.join('D:/PythonProjects/test_dict', 'gen' + str(generation - 2) + '.yaml'), 'w') as f:
        #     yaml.dump(mutate_info_dict, f)
        # test

        mutate_info_dict.clear()
        filename_to_model_dict.clear()
        with alive_bar(last_layer_count, bar='smooth',
                       title=self.seed_model.net_name + '~generation' + str(generation - 1) + '~test run') as bar:
            for i in range(last_layer_count):
                if model_queue.empty():
                    break
                temp_model: Model = model_queue.get()
                file_name = temp_model.model.assemble_file(generation=temp_model.generation, index=temp_model.index)
                run_flag, param_flag = run.run_single_model(file_name, temp_model.model.net_name)
                torch.cuda.empty_cache()
                # something else also need to be returned
                bar()
                if run_flag:
                    temp_model.param_count = param_flag
                    mutate_info = temp_model.mutate_info
                    if mutate_info not in mutate_info_dict.keys():
                        mutate_info_dict[mutate_info] = {}
                    mutate_info_dict[temp_model.mutate_info][file_name] = param_flag
                    filename_to_model_dict[file_name] = copy.deepcopy(temp_model)
                else:
                    pass

        # now, this model list is for all models that has passed 'run', next they will get cut and trained. and
        # we have file_name to param count dict mutate_info_dict to cut, and a filename_to_model_dict to get Models.
        file_list: list[str] = []
        for mutate_info in mutate_info_dict.keys():
            now_dict = mutate_info_dict[mutate_info]
            if len(now_dict.keys()) == 0:
                continue
            elif len(now_dict.keys()) == 1:
                f = list(now_dict.keys())[0]  # file name
                model = filename_to_model_dict[f]
                file_list.append(f)
            else:
                n = len(now_dict.keys())
                if n % 2 == 1:
                    n = int(n / 2) + 1
                else:
                    n = int(n / 2)
                now_dict = self.sort_dict_by_value(now_dict)
                for i in range(n):
                    file_list.append(list(now_dict.keys())[i])

        # now, we have a file list that filled with file names of models that have been cut,
        # next, they will be trained and judged

        with alive_bar(len(file_list), bar='smooth',
                       title=self.seed_model.net_name + '~generation' + str(generation - 1) + '~test train') as bar:
            for file_name in file_list:
                train_flag = run.train_single_model(file_name, self.seed_model.net_name)
                torch.cuda.empty_cache()
                bar()
                if train_flag:
                    model_list.append(filename_to_model_dict[file_name])
                else:
                    continue

        self.dfc(os.path.join(file_paths.MUTATED_MODEL_PATH, self.seed_model.net_name))


if __name__ == '__main__':
    a = Assembler_Complex('BiLSTM')
    a.assemble_code_tree()

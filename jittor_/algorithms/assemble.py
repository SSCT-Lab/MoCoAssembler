import copy
import os
import queue

import file_paths
import simple_model_split as sms
import mutate


class Assembler:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.splited_model_dict = sms.split_model(os.path.join(file_paths.SIMPLE_MODEL_PATH, (model_name + '.py')))
        self.name_to_api_dict = self.analyse_dict(self.splited_model_dict['part2'])
        self.m = mutate.Mutator()
        self.n = 2

        return

    def assemble_code_tree(self):
        model_name = self.model_name
        n = self.n
        original_dict = {'super': self.splited_model_dict['super'], 'part1': self.splited_model_dict['part1'],
                    'execute': self.splited_model_dict['execute'], 'return': self.splited_model_dict['return'],
                    'part4': self.splited_model_dict['part4'], 'part2': [], 'part3': []}
        part3_list = self.splited_model_dict['part3']

        generation = 1


        dic_queue = queue.Queue()
        dic_queue.put(original_dict)



        for sentence in part3_list:
            if 'self.' not in sentence:
                new_queue = queue.Queue()
                while not dic_queue.empty():
                    temp = dic_queue.get()
                    temp['part3'].append(sentence)
                    new_queue.put(temp)
                dic_queue = new_queue
                continue


            last_layer_count = self.n ** (generation - 1)
            # 上一代模型出队，并写入文件
            dic_list = []
            for i in range(last_layer_count):
                temp_dict = dic_queue.get()
                file_name = model_name + '-' + str(generation-1) + '-' + str(i+1)
                self.assemble_dictionary_in_file(temp_dict, file_name)
                dic_list.append(temp_dict)

            for i in range(last_layer_count):
                now_origin = dic_list[i]
                for j in range(self.n):
                    index = 3 * i + j + 1
                    temp = copy.deepcopy(now_origin)
                    now_line = sentence.replace(' ', '').replace('\n', '')
                    name = now_line.split('.')[1].split('(')[0]
                    layer_declaration = self.name_to_api_dict[name]
                    layer = self.m.api_mutate(layer_declaration)
                    part_2_line = '        self.' + name + ' = ' + layer + '\n'
                    temp['part2'].append(part_2_line)
                    temp['part3'].append(sentence)
                    dic_queue.put(copy.deepcopy(temp))




            generation = generation+1

        last_layer_count = self.n ** (generation - 1)
        # 最后一代模型出队，并写入文件
        for i in range(last_layer_count):
            temp_dict = dic_queue.get()
            file_name = model_name + '-' + str(generation - 1) + '-' + str(i)
            self.assemble_dictionary_in_file(temp_dict, file_name)


    def assemble_dictionary_in_file(self, model_dict: dict, file_name: str) -> str:
        seed_model_name = file_name.split('_')[0]
        path = os.path.join(file_paths.MUTATED_MODEL_PATH, file_name) + '.py'
        f = open(path, 'w')
        # part 1 (import, class declaration)
        for str in model_dict['part1']:
            f.write(str)
        # super
        f.write(model_dict['super'])
        # part 2 (layer declaration)
        for str in model_dict['part2']:
            f.write(str)
        # execute
        f.write(model_dict['execute'])
        # part 3 (forward/execute)
        for str in model_dict['part3']:
            f.write(str)
        # return
        f.write(model_dict['return'])
        # part 4 (main)
        for str in model_dict['part4']:
            f.write(str)
        f.close()
        print(file_name + 'successfully assembled')
        return path

    def analyse_dict(self, part_2_list: list) -> dict:
        # return {name -> api}
        result = {}
        for sentence in part_2_list:
            now_line: str = sentence.replace(' ','').replace('\n','')
            if 'self.' in now_line and 'jittor.nn.' in sentence:
                name = now_line.split('=')[0].split('.')[1]
                api = now_line.split('=')[1]
                result[name] = api
        return result

    def set_n(self, val: int):
        self.n = val


if __name__ == '__main__':
    a = Assembler('lenet')
    a.assemble_code_tree()
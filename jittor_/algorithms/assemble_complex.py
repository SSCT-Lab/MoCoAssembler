import copy
import queue

import mutate
from depart import Departed_Model, Single_Model
from complex_models import get_seed_model
import shape_fix
import run
import random

class Model:
    def __init__(self, dm: Departed_Model, g: int, i: int):
        self.model: Departed_Model = dm
        self.generation: int = g
        self.index: int = i


class Assembler_Complex:
    def __init__(self, model_name):
        self.seed_model: Departed_Model = get_seed_model(model_name)
        self.mutator = mutate.Mutator()
        self.shape_fixer = shape_fix.ShapeFixer()
        self.n = 5

    def set_n(self, val: int):
        self.n = val

    def assemble_code_tree(self):
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
                if 'jittor.nn.' in dec:
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

        for sentence in execute_list:
            if model_queue.empty():
                break

            # no need to change, add this in all models in queue
            if 'self.' not in sentence or sentence.startswith('        if') or sentence.startswith('            ') or sentence.startswith('        for'):
                new_queue = queue.Queue()
                while not model_queue.empty():
                    temp: Model = model_queue.get()
                    temp.model.main_model.execute.append(sentence)
                    new_queue.put(temp)
                model_queue = new_queue
                continue

            model_list: list[Model] = []
            next_layer_count = 0
            for i in range(last_layer_count):
                temp_model = model_queue.get()
                file_name = temp_model.model.assemble_file(generation=temp_model.generation, index=temp_model.index)
                run_flag = run.run_single_model(file_name)
                if run_flag:
                    model_list.append(temp_model)
                else:
                    pass

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
                            temp.model.main_model.declaration[name] = new_dec
                            temp.model.main_model.execute.append(sentence)
                        else:
                            if 'jittor.nn.' in dec:
                                new_dec, mut = self.mutator.api_mutate(dec)
                                temp.model.main_model.declaration[name] = new_dec
                                # 6.4尝试增加shape fix
                                if mut != 'no mutate' and 'GRU' not in new_dec and 'LSTM' not in new_dec:
                                    shape_fix_sentence = self.shape_fixer.get_shape_fix_sentence(
                                        'self.' + name + ' = ' + new_dec
                                    )
                                    if random.randint(1,10) > 3:
                                        temp.model.main_model.execute.append(shape_fix_sentence)
                                # no shape fix
                                temp.model.main_model.execute.append(sentence)
                            else:
                                for block_name in block_list:
                                    if block_name in dec:
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

            generation = generation + 1
            last_layer_count = next_layer_count

        for i in range(last_layer_count):
            if model_queue.empty():
                break
            temp = model_queue.get()
            file_name = temp.model.assemble_file(temp.generation, temp.index)
            run.run_single_model(file_name)


if __name__ == '__main__':
    a = Assembler_Complex('alexnet')
    a.assemble_code_tree()

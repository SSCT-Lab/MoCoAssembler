import os.path

from MoCo import MoCo
from depart import Departed_Model, Single_Model
import complex_models
import simple_model_split
import file_paths
import assemble
import assemble_complex
import os
import shutil


def delete_folder_contents(folder_path=file_paths.MUTATED_MODEL_PATH):
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


class MoCoJT(MoCo):

    def __init__(self, model_name: str):
        self.model = None
        self.model_name = model_name

    def depart(self):
        pass

    def mutate(self):
        pass

    def generate_model(self):
        complex_model_list = []
        simple_model_list = []
        for n in ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'nasnet', 'LSTM', 'BiLSTM', 'GRU']:
            complex_model_list.append(n)
        for n in ['testnet', 'lenet', 'alexnet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19', 'densenet']:
            simple_model_list.append(n)
        for i in range(1234):
            for model in simple_model_list:
                a = assemble_complex.Assembler_Complex(model)
                a.assemble_code_tree()
                del a
                delete_folder_contents(file_paths.MUTATED_MODEL_PATH)
                source = os.path.join(file_paths.MAIN_PATH, 'log.txt')
                new_source = os.path.join(file_paths.MAIN_PATH, 'log' + model + '_' + str(i) + '.txt')
                os.rename(source, new_source)
            for model in complex_model_list:
                a = assemble_complex.Assembler_Complex(model)
                a.assemble_code_tree()
                del a
                delete_folder_contents(file_paths.MUTATED_MODEL_PATH)
                source = os.path.join(file_paths.MAIN_PATH, 'log.txt')
                new_source = os.path.join(file_paths.MAIN_PATH, 'log' + model + '_' + str(i) + '.txt')
                os.rename(source, new_source)

    def get_function(self, line: str):
        pass

    def get_params(self, line: str):
        pass

    def generate_param_line(self, line: str, params_dict: dict) -> str:
        pass

    def random_param(self, data) -> str:
        pass

    def mutate_on_parma(self, line: str, func_file: str) -> str:
        pass

    def mutate_on_function(self, line: str, func_file: str) -> str:
        pass

    def mutate_on_module(self, function: str, Inception: dict) -> str:
        pass

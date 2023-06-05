import os.path
import file_paths

from depart import Departed_Model
from depart import Single_Model
import depart


def get_seed_model(model_name: str) -> Departed_Model:
    model_name_list = ['ResNet18', 'ResNet50', 'nasnet', 'InceptionV3', 'xception', 'testnet',
                       'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19']
    assert model_name in model_name_list, 'no such model'

    # simple models

    if model_name in ['alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19']:
        return Departed_Model(model_name)

    # testnet
    if model_name == 'testnet':
        return Departed_Model('testnet')

    # ResNet18
    if model_name == 'ResNet18':
        return Departed_Model('ResNet18')

    # ResNet50
    if model_name == 'ResNet50':
        return Departed_Model('ResNet50')

    # InceptionV3
    if model_name == 'InceptionV3':
        return Departed_Model('InceptionV3')

    if model_name == 'xception':
        result = Departed_Model('xception')
        result.get_model_from_file('xception_for_read')
        with open(os.path.join(file_paths.MODEL_PATH, 'xception.py'), 'r') as f:
            lines = f.readlines()
            f.close()
        result.main_model.execute_sentence = depart.connect_str_list(lines[44:67]) + result.main_model.execute_sentence
        return result

    # NasNet
    if model_name == 'nasnet':
        with open(os.path.join(file_paths.MODEL_PATH, 'nasnet.py'), 'r') as f:
            lines = f.readlines()
            f.close()
        nasnet = Departed_Model('nasnet')
        nasnet.get_model_from_file('nasnet')
        nasnet.block_dict['Bottleneck'].init_sentence += depart.connect_str_list(lines[139:143])
        nasnet.block_dict['BasicBlock'].init_sentence += depart.connect_str_list(lines[102:109])
        nasnet.main_model.init_sentence += depart.connect_str_list(lines[10:19])
        nasnet.main_model.other_parts = depart.connect_str_list(lines[35:50]) + nasnet.main_model.other_parts
        return nasnet

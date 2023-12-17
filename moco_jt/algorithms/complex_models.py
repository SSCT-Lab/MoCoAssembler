import os.path
import file_paths

from depart import Departed_Model
from depart import Single_Model
import depart


def get_seed_model(model_name: str) -> Departed_Model:
    model_name_list = ['resnet18',
                      'alexnet', 'LeNet', 'mobilenet', 'squeezenet', 'vgg19',
                      'LSTM', 'googlenet', "pointnet"]
    assert model_name in model_name_list, 'no such model'

    # simple models

    if model_name in ['alexnet', 'LeNet', 'mobilenet', 'squeezenet', 'vgg19',
                      'LSTM', 'googlenet', "pointnet"]:
        return Departed_Model(model_name)

    # ResNet18
    if model_name == 'resnet18':
        return Departed_Model('resnet18')

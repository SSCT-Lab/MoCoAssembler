import random

import moco_jt

from assemble_complex_boundary import Assembler_Complex
import argparse
import os
import file_paths

if __name__ == '__main__':
    MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
                  'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                  'densenet', 'BiLSTM', 'LSTM', 'GRU', 'LOOP', 'googlenet']

    # params: ========
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = 5
    TSF = 1

    assert MODEL in MODEL_LIST
    assert TSF in [0, 1]
    moco_jt.flags.use_cuda = 1
    skip_list = ['LOOP', 'testnet']

    if MODEL == 'LOOP':
        if True:
            skip_list = ['LOOP', 'testnet']
            for model in MODEL_LIST:
                os.environ['TRAIN_STOP_FLAG'] = str(TSF)
                if model in skip_list:
                    continue
                try:
                    a = Assembler_Complex(model)
                    a.set_n(N)
                    a.assemble_code_tree()
                    os.rename(os.path.join(file_paths.LOG_PATH, model, 'log.txt'),
                              os.path.join(file_paths.LOG_PATH, model, 'log' + str(random.randint(10000, 20000))
                                           + '.txt'))
                except Exception:
                    skip_list.append(model)
                    continue
    # ================

    # run
    else:
        a = Assembler_Complex(MODEL)
        a.set_n(N)
        a.assemble_code_tree()

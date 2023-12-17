import random
from assemble_complex_boundary import Assembler_Complex
import argparse
import os
import file_paths

if __name__ == '__main__':
    MODEL_LIST = ['resnet18',
                  'alexnet', 'LeNet', 'mobilenet', 'squeezenet', 'vgg19',
                  'LSTM', 'googlenet', "pointnet"]

    # params: ========
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = 5

    assert MODEL in MODEL_LIST
    skip_list = ['LOOP', 'testnet']

    if MODEL == 'LOOP':
        if True:
            skip_list = ['LOOP', 'testnet']
            for model in MODEL_LIST:
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

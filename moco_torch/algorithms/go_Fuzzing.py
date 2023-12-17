import os
import traceback

import file_paths
from assemble_complex import Assembler_Complex
import argparse

if __name__ == '__main__':
    MODEL_LIST = ['resnet18',
                  'alexnet', 'LeNet', 'mobilenet', 'squeezenet', 'vgg19',
                  'LSTM', 'googlenet', "pointnet"]

    # params: ========
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')
    parser.add_argument('--N', type=int, default=3, help='N value')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = args.N

    skip_list = ['LOOP', 'testnet']

    assert MODEL in MODEL_LIST

    if MODEL == 'LOOP':
        for i in range(5):
            for model in MODEL_LIST:
                if model in skip_list:
                    continue
                try:
                    a = Assembler_Complex(model)
                    a.set_n(N)
                    a.assemble_code_tree()
                    os.rename(os.path.join(file_paths.LOG_PATH, model, 'log.txt'),
                              os.path.join(file_paths.LOG_PATH, model, 'log' + str(i) + '.txt'))
                except Exception:
                    error = str(traceback.format_exc())
                    f = open(os.path.join(file_paths.MAIN_PATH, 'report.txt'), 'a', encoding='utf-8')
                    f.write(error)
                    f.close()
                    skip_list.append(model)
                    continue

    # ================

    # run
    else:
        a = Assembler_Complex(MODEL)
        a.set_n(N)
        a.assemble_code_tree()

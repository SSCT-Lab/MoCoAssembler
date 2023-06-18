import os
import file_paths
from assemble_complex import Assembler_Complex
import argparse

if __name__ == '__main__':
    MODEL_LIST = ['ResNet18', 'ResNet50', 'nasnet', 'InceptionV3', 'xception', 'testnet',
                  'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                  'densenet', 'LSTM', 'GRU', 'LOOP']

    # params: ========
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')
    parser.add_argument('--N', type=int, default=3, help='N value')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = args.N

    assert MODEL in MODEL_LIST

    if MODEL == 'LOOP':
        for i in range(100):
            for model in MODEL_LIST:
                if model == 'LOOP':
                    continue
                try:
                    a = Assembler_Complex(model)
                    a.set_n(N)
                    a.assemble_code_tree()
                    os.rename(os.path.join(file_paths.LOG_PATH, model, 'log.txt'),
                              os.path.join(file_paths.LOG_PATH, model, 'log' + str(i) + '.txt'))
                except Exception:
                    continue

    # ================

    # run
    a = Assembler_Complex(MODEL)
    a.set_n(N)
    a.assemble_code_tree()

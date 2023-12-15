import moco_jt

from assemble_complex_v2 import Assembler_Complex
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
    parser.add_argument('--N', type=int, default=2, help='N value')
    parser.add_argument('--TSF', type=int, default=0, help='Train Stop Flag')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = args.N
    TSF = args.TSF

    assert MODEL in MODEL_LIST
    assert TSF in [0, 1]
    if TSF == 1:
        print("*** train closed ***")
    else:
        print("*** train opened ***")
    moco_jt.flags.use_cuda = 1
    skip_list = ['LOOP', 'testnet']


    if MODEL == 'LOOP':
        for i in range(5):
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
                              os.path.join(file_paths.LOG_PATH, model, 'log' + str(i) + '.txt'))
                except Exception:
                    skip_list.append(model)
                    continue
    # ================

    # run
    else:
        os.environ['TRAIN_STOP_FLAG'] = str(TSF)
        a = Assembler_Complex(MODEL)
        a.set_n(N)
        a.assemble_code_tree()


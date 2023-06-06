from assemble_complex import Assembler_Complex
import argparse

if __name__ == '__main__':
    MODEL_LIST = ['ResNet18', 'ResNet50', 'nasnet', 'InceptionV3', 'xception', 'testnet',
                       'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                       'densenet', 'BiLSTM', 'LSTM', 'GRU']

    # params: ========
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='testnet', help='Model name')
    parser.add_argument('--N', type=int, default=2, help='N value')

    args = parser.parse_args()

    MODEL = args.MODEL
    N = args.N

    assert MODEL in MODEL_LIST
    # ================

    # run
    a = Assembler_Complex(MODEL)
    a.set_n(N)
    a.assemble_code_tree()


from pathlib import Path

import sys
sys.path.append(Path.cwd().parent.parent.__str__())

import argparse

from tensorflow_.src.mutate_tf import MoCoTF

simple_model = ["lenet",
                "alexnet",
                "vgg16",
                "vgg19",
                "mobilenet"
                ]

complex_model = ["googlenet",
                 "resnet18",
                 "resnet50",
                 "squeezenet",
                 "xception",
                 "densenet",
                 "inceptionv3",
                 "lstm",
                 "bilstm",
                 "gru"
                ]
def train(model, mutate_times):
    mocoTf = MoCoTF(model, mutate_times)

    # depart one model
    if (mocoTf.res_model_dir / mocoTf.template_file_name).exists():
        print(model + " decomposition files exist.")
        pass
    else:
        print(model + " decomposition file does not exist, we will create it……")
        mocoTf.depart()
        print(model + " decomposition complete.")

    # generate new model list
    mocoTf.mutate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--train_simple',
                        type=bool,
                        default=False,
                        required=False,
                        help="whether train simple seed model")
    parser.add_argument('--train_complex',
                        type=bool,
                        default=False,
                        required=False,
                        help="whether train complex seed model")
    parser.add_argument('--train_all',
                        type=bool,
                        default=False,
                        required=False,
                        help="whether train all seed model")
    parser.add_argument('--model_name',
                        type=str,
                        default="lenet",
                        required=False,
                        help="model name")
    parser.add_argument('--mutate_times',
                        type=int,
                        default=2,
                        required=True,
                        help="mutate_times")

    args = parser.parse_args()

    if args.train_simple:
        for model in simple_model:
            print(model + " mutation start.")
            train(model, args.mutate_times)
            print(model + " mutation complete.")
    elif args.train_complex:
        for model in complex_model:
            print(model + " mutation start.")
            train(model, args.mutate_times)
            print(model + " mutation complete.")
    elif args.train_all:
        for model in simple_model + simple_model:
            print(model + " mutation start.")
            train(model, args.mutate_times)
            print(model + " mutation complete.")
    else:
        train(args.model_name, args.mutate_times)
        print(args.model_name + " mutation complete.")

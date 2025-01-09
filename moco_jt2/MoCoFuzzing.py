from Tools.Assembler import Assembler

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default="lenet",
                        required=False)
    parser.add_argument('--n',
                        type=int,
                        default=3,
                        required=False)
    parser.add_argument('--max',
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument('--output',
                        type=str,
                        default="lenet",
                        required=False)

    args = parser.parse_args()

    a = Assembler(args.model, args.n, args.max, args.output)
    a.start()

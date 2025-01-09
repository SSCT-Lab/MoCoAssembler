from Tools.Assembler import Bssembler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default="lenet",
                        required=False,
                        help="Model name")
    parser.add_argument('--output',
                        type=str,
                        default="lenet",
                        required=False)

    args = parser.parse_args()

    b = Bssembler(args.model, args.output)
    b.start()

from Tools.Boundary import Boundary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default="lenet",
                        required=False,
                        help="Model name")
    parser.add_argument('--fuzzing',
                        type=int,
                        default=3,
                        required=False)
    parser.add_argument('--output',
                        type=str,
                        default="lenet",
                        required=False)

    args = parser.parse_args()

    b = Boundary(args.model, args.fuzzing, args.output)
    b.start()

import time
import argparse
import os
from Tools.Assembler import Assembler


if __name__ == "__main__":
    os.makedirs("./Output", exist_ok=True)
    parser = argparse.ArgumentParser(description="moco mxnet")
    parser.add_argument('-s', '--seed', type=str, default='lenet',
                        help='The seed for the operation (default: lenet)')
    args = parser.parse_args()
    a = Assembler(args.seed, 1, 100, f"{args.seed}_{str(time.time())}")
    a.start()

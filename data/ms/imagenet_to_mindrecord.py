# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""imagenet_to_mindrecord.py"""
import argparse
import os
import time

from mindspore.mindrecord import ImageNetToMR


def main():
    parser = argparse.ArgumentParser(description="ImageNet to MindRecord")
    parser.add_argument('--data_url', default="./imagenet", help='Location of data.')
    args = parser.parse_args()
    image_dir_train = os.path.join(args.data_url, "train")
    image_dir_val = os.path.join(args.data_url, "val")
    map_file = "labels.txt"
    get_map_file(image_dir_train, image_dir_val, map_file)

    imagenet_dir = f'{os.sep}'.join(args.data_url.split(os.sep)[:-1])
    imagenet_dir = os.path.join(imagenet_dir, "imagenet_mind")
    os.makedirs(imagenet_dir, exist_ok=True)

    start = time.time()
    train_destination = os.path.join(imagenet_dir, "imagenet_train.mindrecord")
    print(f"=> Begin to process train dataset to {train_destination}")
    train = ImageNetToMR(map_file, image_dir_train, train_destination, partition_number=8)
    train.run()
    end = int(time.time() - start)
    print(f"=> for train dataset: {end // 3600}h {end // 60 % 60}m {end % 3600}s")

    start = time.time()
    val_destination = os.path.join(imagenet_dir, "imagenet_val.mindrecord")
    print(f"=> Begin to process val dataset to {val_destination}")
    val = ImageNetToMR(map_file, image_dir_val, val_destination, partition_number=8)
    val.run()
    end = int(time.time() - start)
    print(f"=> for val dataset: {end // 3600}h {end // 60 % 60}m {end % 3600}s")

    if os.path.exists(map_file):
        os.remove(map_file)


def get_map_file(train_dir, val_dir, map_file="labels.txt"):
    train_dir_list = sorted(os.listdir(train_dir))
    val_dir_list = sorted(os.listdir(val_dir))
    assert train_dir_list == val_dir_list, "train class must be equal to val class"
    with open(map_file, 'w') as f:
        for index, dir in enumerate(train_dir_list):
            f.writelines(f"{dir} {index}\n")


if __name__ == '__main__':
    main()

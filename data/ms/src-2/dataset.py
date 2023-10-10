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
"""generator dataset for train,eval and test"""
from PIL import Image
import os
import os.path
import numpy as np
from src.transforms import *
from numpy.random import randint

import mindspore.dataset as ds

class VideoRecord:
    def __init__(self, root_path, row):
        self.root_path = root_path
        self._data = row

    @property
    def path(self):
        return os.path.join(self.root_path, self._data[0])

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

def process_input(image, input_size, roll, input_mean, input_std, div, test_crops, scale_size, modality, test_mode=0):
    if test_mode == 0:
        scales = [1, .875, .75] if modality!="RGB" else [1, .875, .75, .66]
        transform_1 = GroupMultiScaleCrop(input_size, scales)
        transform_2 = GroupRandomHorizontalFlip(is_flow=modality=='Flow')
        transform_3 = Stack(roll=roll)
        transform_4 = ToTorchFormatTensor(div=div)
        if modality != 'RGBDiff':
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()
        transform_5 = normalize
        img = transform_1(image)
        img = transform_2(img)
        img = transform_3(img)
        img = transform_4(img)
        img = transform_5(img)
    else:
        if test_crops == 1:
            transform_1 = GroupScale(int(scale_size))
            transform_2 = GroupCenterCrop(input_size)
        elif test_crops == 10:
            transform_1 = GroupOverSample(input_size, scale_size)
            transform_2 = None
        else:
            raise ValueError("Only 1 and 10 crops are supported while we got {}".format(test_crops))

        transform_3 = Stack(roll=roll)
        transform_4 = ToTorchFormatTensor(div=div)
        if test_mode==1:
            if modality != 'RGBDiff':
                normalize = GroupNormalize(input_mean, input_std)
            else:
                normalize = IdentityTransform()
            transform_5 = normalize
        else:
            transform_5 = GroupNormalize(input_mean, input_std)
        img = transform_1(image)
        if transform_2:
            img = transform_2(img)
        img = transform_3(img)
        img = transform_4(img)
        img = transform_5(img)

    return img

class TSNDataSet:
    def __init__(self, root_path, list_file, input_size, roll, input_mean, input_std, div, test_crops, scale_size,
                 num_segments=3, new_length=1, modality='RGB', image_tmpl='img_{:05d}.jpg', random_shift=True, test_mode=False):
        self.input_size = input_size
        self.roll = roll
        self.input_mean = input_mean
        self.input_std = input_std
        self.div = div
        self.test_crops = test_crops
        self.scale_size = scale_size

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff
        self.video_list = []
        with open(self.list_file) as f:   
            for line in f:
                self.video_list.append(VideoRecord(self.root_path, line.strip().split(' ')))

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            with open(os.path.join(directory, self.image_tmpl.format(idx)), 'rb') as f:
                return [Image.open(f).convert('RGB')]
        elif self.modality == 'Flow':
            with open(os.path.join(directory, self.image_tmpl.format('x', idx)), 'rb') as f:
                x_img = Image.open(f).convert('L')
            with open(os.path.join(directory, self.image_tmpl.format('y', idx)), 'rb') as f:
                y_img = Image.open(f).convert('L')
            result = [x_img, y_img]
            return result

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):

        record = self.video_list[index]

        if self.test_mode!=2:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = process_input(image=images, input_size=self.input_size, roll=self.roll, input_mean=self.input_mean,\
             input_std=self.input_std, div=self.div, test_crops=self.test_crops, scale_size=self.scale_size, modality=self.modality, test_mode=self.test_mode)

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
        #return 16

class Get_Diff:
    """Multi scale transform."""
    def __init__(self, modality, new_length, num_segments, keep_rgb=False):
        self.modality = modality
        self.new_length = new_length
        self.keep_rgb = keep_rgb
        self.reverse = list(range(self.new_length, 0, -1))
        
        self.num_segments = num_segments
        self.input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2

    def __call__(self, input, label, batch_info):

        input = np.array(input)
        input_view = input.reshape((-1, self.num_segments, self.new_length + 1, self.input_c,) + input.shape[2:])
        if self.keep_rgb:
            new_data = input_view.copy()
        else:
            new_data = input_view[:, :, 1:, :, :, :].copy()

        for x in self.reverse:
            if self.keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return np.array(new_data), np.array(label)


def create_dataset(root_path, list_file, batch_size, input_size, roll, input_mean, input_std, div, test_crops, scale_size, num_segments=3, new_length=1,\
 modality='RGB', image_tmpl='img_{:05d}.jpg', random_shift=True, test_mode=0, run_distribute=False, worker=1, num_shards=1, shard_id=0):
    """
    :param test_mode: 0 train; 1 eval; 2 test;
    :return: list
    """
    data = TSNDataSet(root_path, list_file, input_size, roll, input_mean, input_std, div, test_crops, scale_size, num_segments=num_segments, new_length=new_length, modality=modality,\
     image_tmpl=image_tmpl, random_shift=random_shift, test_mode=test_mode)
    if test_mode:
        shuffle = False
    else:
        shuffle = True
    if run_distribute:
        dataset = ds.GeneratorDataset(data, column_names=["input", "label"], num_parallel_workers=worker, shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
    else:
        dataset = ds.GeneratorDataset(data, column_names=["input", "label"], num_parallel_workers=worker, shuffle=shuffle)
    ds.config.set_enable_shared_mem(False)
    if modality=="RGBDiff" and test_mode==0:
        getdiff = Get_Diff(modality, new_length, num_segments)
        dataset = dataset.batch(batch_size=batch_size, per_batch_map=getdiff, input_columns=["input", "label"],
                          num_parallel_workers=min(32, worker))
    else:
        dataset = dataset.batch(batch_size)

    return dataset

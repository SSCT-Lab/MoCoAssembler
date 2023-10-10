#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : dataset.py
#   Author      : VXallset
#   Created date: 2021-11-16
#   Description : dataset providerfor segmentation
#
#================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from core.config import cfg


class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type='train'):
        self.img_path = cfg.TRAIN.IMG_PATH if dataset_type == 'train' else cfg.TEST.IMG_PATH
        self.label_path = cfg.TRAIN.LABEL_PATH if dataset_type == 'train' else cfg.TEST.LABEL_PATH
        #self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.img_data = []
        self.label_data = []
        self.output_img_shape = [1080, 1920]
        self.preload()
        self.img_num = len(self.label_data)
        self.load_sequence = [i for i in range(self.img_num)]
        self.num_batches = self.img_num // self.batch_size
        self.batch_count = 0

    def preload(self):
        imgs = os.listdir(self.img_path)
        segmentations = os.listdir(self.label_path)

        for segmentation in segmentations:
            if segmentation in imgs:
                self.img_data.append(segmentation)
                self.label_data.append(segmentation)
            else:
                basename, _ = os.path.splitext(segmentation)
                posible_subfixs = ['.jpg', '.jpeg', '.png', ',PNG', '.JPEG', '.JPG']
                for subfix in posible_subfixs:
                    if basename + subfix in imgs:
                        self.img_data.append(basename + subfix)
                        self.label_data.append(segmentation)
                        break
        print("Finished building dataset, pre-loaded {} images.".format(len(self.img_data)))


    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):

            batch_image = np.zeros((self.batch_size, self.output_img_shape[0], self.output_img_shape[1], 3), dtype=np.float32)
            batch_label = np.zeros((self.batch_size, self.output_img_shape[0], self.output_img_shape[1], 1), dtype=np.float32)


            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.img_num:
                        index -= self.img_num
                    image, label = self.load_img_and_label(index)

                    batch_image[num, :, :, :] = image
                    batch_label[num, :, :, :] = label

                    num += 1
                self.batch_count += 1

                return batch_image, batch_label
            else:
                self.batch_count = 0
                np.random.shuffle(self.load_sequence)
                raise StopIteration

    def load_img_and_label(self, index):
        _index = self.load_sequence[index]

        image_path = os.path.join(self.img_path, self.img_data[_index])
        label_path = os.path.join(self.label_path, self.label_data[_index])
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        if len(label.shape) == 3:
            label = np.expand_dims(label[:, :, 0], axis=-1)

        if self.data_aug:
            image, label = self.random_horizontal_flip(np.copy(image), np.copy(label))

        image = self.image_preporcess(np.copy(image))
        label = self.image_preporcess(np.copy(label))

        return image, label

    @classmethod
    def image_preporcess(cls, image, target_size=(1080, 1920)):

        ih, iw    = target_size
        h,  w, c  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        if c == 1:
            image_resized =np.expand_dims(image_resized, axis=-1)

        image_paded = np.full(shape=[ih, iw, c], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        return image_paded

    @classmethod
    def image_postporcess(cls, image, target_size=(1080, 1920, 3)):
        ih, iw = image.shape
        h,  w, c = target_size

        scale = min(iw/w, ih/h)
        nw, nh = int(scale * w), int(scale * h)

        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        _temp_img = image[dh:nh+dh, dw:nw+dw]

        result_image = np.repeat(np.expand_dims(_temp_img, axis=-1), 3, axis=-1) * 255
        return result_image


    def random_horizontal_flip(self, image, label):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]

        return image, label

    def __len__(self):
        return self.num_batches





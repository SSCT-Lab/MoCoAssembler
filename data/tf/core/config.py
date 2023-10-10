#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : config.py
#   Author      : VXallset
#   Created date: 2021-11-16
#   Description : Config file for BiSeNet_V2
#
#================================================================

from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Solver options
__C.SOLVER = edict()

__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.ENABLE_OHEM = True
__C.SOLVER.LOSS_TYPE = 'cross_entropy'
__C.SOLVER.OHEM_SCORE_THRESH = 0.65
__C.SOLVER.OHEM_MIN_SAMPLE_NUMS = 262144

# Dataset options
__C.DATASET = edict()

__C.DATASET.NUM_CLASSES = 2


# Model options
__C.MODEL = edict()

__C.MODEL.GE_EXPAND_RATIO = 6
__C.MODEL.SEMANTIC_CHANNEL_LAMBDA = 0.25
__C.MODEL.SEGHEAD_CHANNEL_EXPAND_RATIO = 2


# Train options
__C.TRAIN = edict()

__C.TRAIN.IMG_PATH = "C:\\home\\dataset\\track_segmentation\\track_1\\processed\\track_ori_imgs"
__C.TRAIN.LABEL_PATH = "C:\\home\\dataset\\track_segmentation\\track_1\\processed\\track_labels_imgs"
__C.TRAIN.BATCH_SIZE = 4

__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.EPOCHS = 30



# TEST options
__C.TEST = edict()
__C.TEST.MODEL_PATH = "../core/model/bisenet_v2_epoch3"

__C.TEST.IMG_PATH = "C:\\home\\dataset\\track_segmentation\\track_1\\processed\\track_ori_imgs"
__C.TEST.LABEL_PATH = "C:\\home\\dataset\\track_segmentation\\track_1\\processed\\track_labels_imgs"
__C.TEST.BATCH_SIZE = 1

__C.TEST.DATA_AUG = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"




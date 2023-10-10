# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
dataloader.py
~~~~~~~~~~~~~

A class responsibe for load batch of data for training and eval.
"""
import os
import h5py
import json
import random
import numpy as np

# paddle
import paddle
from paddle.io import Dataset, DataLoader

class CaptionDataset(Dataset):
    def __init__(self, opt, idx, ix_to_word, images, labels, label_start_ix, label_end_ix):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.feat_dir = opt.feat_dir

        self.idx = idx
        self.ix_to_word = ix_to_word
        self.vocab_size = len(self.ix_to_word)
        self.images = images

        self.labels = labels
        self.seq_length = labels.shape[-1]
        self.label_start_ix = label_start_ix
        self.label_end_ix = label_end_ix

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def fetch_feats(self, ix):
        feats = np.load(os.path.join(self.feat_dir, str(self.images[ix]['id']) + '.npz'))
        att_feat = feats['att_feat']
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])  # [num_object, feat_dim]
        fc_feat = feats['fc_feat']

        return fc_feat, att_feat

    def fetch_seqs(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.labels[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.labels[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def __getitem__(self, index):
        # global ix in the self.images
        ix = self.idx[index]

        # fetch feats
        fc_feat, att_feat = self.fetch_feats(ix)

        # fetch seqs
        seqs = self.fetch_seqs(ix, self.seq_per_img)
        label = np.zeros((self.seq_per_img, self.seq_length + 2), dtype='int64')
        label[:, 1: self.seq_length+1] = seqs  # 0 is for start and end tokens

        # all captions for this images (used to compute self-critial reward and eval)
        gt = self.labels[self.label_start_ix[ix] - 1: self.label_end_ix[ix]].astype('int32')

        # prepare mask
        mask = np.zeros((self.seq_per_img, self.seq_length + 2))
        for i in range(self.seq_per_img):
            num_nonzeros = (label[i] != 0).sum() + 2
            mask[i, :num_nonzeros] = 1

        # record associated info as well
        info_dict = {}
        info_dict['ix'] = ix
        info_dict['id'] = self.images[ix]['id']
        info_dict['filename'] = self.images[ix]['filename']

        return fc_feat, att_feat, label, mask, gt, info_dict

    def __len__(self):
        return len(self.idx)

def create_collate_fn():
    def collate_fn(dataset):
        gts = []
        infos = []
        tmp = []
        num_objects_list = []
        for fc_feat, att_feat, label, mask, gt, info_dict in dataset:
            gts.append(gt)
            infos.append(info_dict)
            # label: [seq_per_img, seq_length + 2]
            for i in range(label.shape[0]):
                tmp.append([fc_feat, att_feat, label[i], mask[i]])
                num_objects_list.append(att_feat.shape[0])

        fc_feats, att_feats, labels, masks = zip(*tmp)
        fc_feats = np.array(fc_feats, dtype='float32')
        labels = np.array(labels, dtype='int64')
        masks = np.array(masks, dtype='float32')

        # prepare att_mask and att_feats which has varied objects for different images
        max_num_objects = max(num_objects_list)
        att_feat_dim = att_feats[0].shape[-1]
        # [batch_size * seq_per_image, max_num_objects]
        batch_att_masks = np.zeros((len(att_feats), max_num_objects), dtype='float32')
        # [batch_size * seq_per_image, max_num_objects, feat_dim]
        batch_att_feats = np.zeros((len(att_feats), max_num_objects, att_feat_dim), dtype='float32')

        for i, (num_object, att_feat) in enumerate(zip(num_objects_list, att_feats)):
            batch_att_masks[i, :num_object] = 1.0
            batch_att_feats[i, :num_object, :] = att_feat

        # check if att_masks is needed
        if batch_att_masks.sum() == batch_att_masks.size:
            batch_att_masks = None

        return fc_feats, batch_att_feats, batch_att_masks, labels, masks, gts, infos

    return collate_fn

def get_dataloaders(opt):
    """Return dataloaders for training, val and test."""

    # load vocabulary and images from json file
    info = json.load(open(opt.input_json, 'r'))
    ix_to_word = info['ix_to_word']
    images = info['images']
    num_images = len(images)

    # assign images for training, val and test
    split_ix = {'train': [], 'val': [], 'test': []}
    for ix in range(num_images):
        img = images[ix]
        if img['split'] == 'train':
            split_ix['train'].append(ix)
        elif img['split'] == 'val':
            split_ix['val'].append(ix)
        elif img['split'] == 'test':
            split_ix['test'].append(ix)
        else:
            split_ix['train'].append(ix)

    print('assigned %d images to split train' % len(split_ix['train']))
    print('assigned %d images to split val' % len(split_ix['val']))
    print('assigned %d images to split test' % len(split_ix['test']))

    # load encoded captions from h5 file
    h5_label_file = h5py.File(opt.input_label_h5, 'r', driver='core')
    labels = h5_label_file['labels'][:]

    # start and end pointer of each image
    # see propre.py for details
    label_start_ix = h5_label_file['label_start_ix'][:]
    label_end_ix = h5_label_file['label_end_ix'][:]

    # build train/val/test dataloader
    train_dataset = CaptionDataset(opt, split_ix['train'], ix_to_word, images, labels, label_start_ix, label_end_ix)
    val_dataset = CaptionDataset(opt, split_ix['val'], ix_to_word, images, labels, label_start_ix, label_end_ix)
    test_dataset = CaptionDataset(opt, split_ix['test'], ix_to_word, images, labels, label_start_ix, label_end_ix)

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=create_collate_fn())

    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            collate_fn=create_collate_fn())

    test_loader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            collate_fn=create_collate_fn())

    return train_loader, val_loader, test_loader

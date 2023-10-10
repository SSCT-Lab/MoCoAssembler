#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : net_utils.py
#   Author      : VXallset
#   Created date: 2021-11-16
#   Description : utils for net
#
#================================================================

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def maxpooling(input_tensor, kernel_size, stride, padding='SAME'):
    res = tf.keras.layers.MaxPool2D(kernel_size, strides=stride,padding=padding)(input_tensor)
    return res

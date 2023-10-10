#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : block.py
#   Author      : VXallset
#   Created date: 2021-11-16
#   Description : blocks for BiSeNet_V2
#
#================================================================

import tensorflow as tf
from core.net_utils import BatchNormalization, maxpooling


def conv_block(input_tensor, k_size, output_channels, stride,
               padding='SAME', use_bias=False, need_activate=False):
    conv = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=k_size, strides=stride,
                                  padding=padding, use_bias=use_bias)(input_tensor)

    conv = BatchNormalization()(conv)
    if need_activate:
        conv = tf.nn.relu(conv)
    return conv


def _StemBlock(input_tensor, output_channels, padding='SAME'):
    input_tensor = conv_block(input_tensor=input_tensor, k_size=3, output_channels=output_channels,
                              stride=2, padding=padding, use_bias=False, need_activate=True)

    # downsample_branch_left
    branch_left_output = conv_block(input_tensor=input_tensor, k_size=1, output_channels=int(output_channels/2),
                                    stride=1, padding=padding, use_bias=False, need_activate=True)
    branch_left_output = conv_block(input_tensor=branch_left_output, k_size=3, output_channels=output_channels,
                                    stride=2, padding=padding, use_bias=False, need_activate=True)

    # downsample_branch_right
    branch_right_output = maxpooling(input_tensor=input_tensor, kernel_size=3, stride=2)

    result = tf.concat([branch_left_output, branch_right_output], axis=-1)
    result = conv_block(input_tensor=result, k_size=3, output_channels=output_channels,
                        stride=1, padding=padding, use_bias=False, need_activate=True)
    return result


def _ContextEmbedding(input_tensor, padding='SAME'):
    output_channels = input_tensor.get_shape().as_list()[-1]

    result = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    result = BatchNormalization()(result)
    result = conv_block(input_tensor=result, k_size=1, output_channels=output_channels,
                        stride=1, padding=padding, use_bias=False, need_activate=True)

    # fuse features
    result = tf.add(result, input_tensor)
    # final convolution block
    result = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=3, strides=1,
                                    use_bias=False, padding=padding)(result)
    return result

def _GatherExpansion(input_tensor, output_channels=None, padding='SAME', stride=1, e=None):
    if output_channels is None:
        output_channels = input_tensor.get_shape().as_list()[-1]

    if stride == 1:
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        result = conv_block(input_tensor=input_tensor, k_size=3, output_channels=input_tensor_channels,
                            stride=1, padding=padding, use_bias=False, need_activate=True)

        # depthwise conv block
        result = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding=padding, depth_multiplier=e,
                                                 depthwise_initializer=tf.keras.initializers.variance_scaling)(result)
        result = BatchNormalization()(result)
        result = conv_block(input_tensor=result, k_size=1, output_channels=input_tensor_channels,
                            stride=1, padding=padding, use_bias=False, need_activate=False)
        result = tf.add(input_tensor, result)
        result = tf.nn.relu(result)

    elif stride == 2:
        input_tensor_channels = input_tensor.get_shape().as_list()[-1]
        input_proj = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, depth_multiplier=1, padding=padding,
                                                     depthwise_initializer=tf.keras.initializers.variance_scaling)(input_tensor)
        input_proj - BatchNormalization()(input_proj)
        input_proj = conv_block(input_tensor=input_proj, k_size=1, output_channels=output_channels, stride=1,
                                padding=padding, use_bias=False, need_activate=False)
        result = conv_block(input_tensor=input_tensor, k_size=3, output_channels=input_tensor_channels, stride=1,
                            padding=padding, use_bias=False, need_activate=True)
        result = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding=padding, depth_multiplier=e,
                                                 depthwise_initializer=tf.keras.initializers.variance_scaling)(result)
        result = BatchNormalization()(result)
        result = conv_block(input_tensor=result, k_size=1, output_channels=output_channels, stride=1, padding=padding,
                            use_bias=False, need_activate=True)
        result = tf.add(input_proj, result)
        result = tf.nn.relu(result)
    else:
        raise NotImplementedError('No function matched with stride of {}'.format(stride))
    return result


def _GuidedAggregation(detail_input_tensor, semantic_input_tensor, padding='SAME'):
    output_channels = detail_input_tensor.get_shape().as_list()[-1]
    # detail_branch
    detail_branch_remain = tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding=padding, strides=1,
                                                           depthwise_initializer=tf.keras.initializers.variance_scaling)(detail_input_tensor)
    detail_branch_remain = BatchNormalization()(detail_branch_remain)
    detail_branch_remain = tf.keras.layers.Conv2D(kernel_size=1, filters=output_channels, strides=1, padding=padding,
                                                  use_bias=False)(detail_branch_remain)
    detail_branch_downsample = conv_block(input_tensor=detail_input_tensor, k_size=3, output_channels=output_channels,
                                          stride=2, padding=padding, use_bias=False, need_activate=False)
    detail_branch_downsample = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding=padding)(detail_branch_downsample)

    # semantic branch
    semantic_branch_remain = tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding=padding, strides=1,
                                                             depthwise_initializer=tf.keras.initializers.variance_scaling)(semantic_input_tensor)
    semantic_branch_remain = BatchNormalization()(semantic_branch_remain)
    semantic_branch_remain = tf.keras.layers.Conv2D(kernel_size=1, filters=output_channels, strides=1, padding=padding,
                                                    use_bias=False)(semantic_branch_remain)
    semantic_branch_remain = tf.nn.sigmoid(semantic_branch_remain)
    semantic_branch_upsample = conv_block(input_tensor=semantic_input_tensor, k_size=3, output_channels=output_channels,
                                          stride=1, padding=padding, use_bias=False, need_activate=False)
    semantic_branch_upsample = tf.image.resize(semantic_branch_upsample, detail_input_tensor.shape[1:3])
    semantic_branch_upsample = tf.nn.sigmoid(semantic_branch_upsample)

    # aggregation_features
    guided_features_remain = tf.multiply(detail_branch_remain, semantic_branch_upsample)
    guided_features_downsample = tf.multiply(detail_branch_downsample, semantic_branch_remain)
    guided_features_upsample = tf.image.resize(guided_features_downsample, detail_input_tensor.shape[1:3])
    guided_features = tf.add(guided_features_remain, guided_features_upsample)
    guided_features = conv_block(input_tensor=guided_features, k_size=3, output_channels=output_channels, stride=1,
                                 padding=padding, use_bias=False, need_activate=True)
    return guided_features

def _SegmentationHead(input_tensor, upsample_ratio, feature_dims, classes_nums, padding='SAME'):
    input_tensor_size = input_tensor.get_shape().as_list()[1:3]
    output_tensor_size = [int(tmp * upsample_ratio) for tmp in input_tensor_size]
    result = conv_block(input_tensor=input_tensor, k_size=3, output_channels=feature_dims, stride=1, padding=padding,
                        use_bias=False, need_activate=True)
    result = tf.keras.layers.Conv2D(filters=classes_nums, kernel_size=1, padding=padding, strides=1,
                                    use_bias=False)(result)
    result = tf.image.resize(result, output_tensor_size)
    return result

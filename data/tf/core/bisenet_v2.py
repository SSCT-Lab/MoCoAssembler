#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : bisenet_v2.py
#   Author      : VXallset
#   Created date: 2021-11-16
#   Description : BiSeNet_V2
#
#================================================================

import tensorflow as tf
from core.blocks import _StemBlock, _ContextEmbedding, _GatherExpansion, _GuidedAggregation, _SegmentationHead
from core.blocks import conv_block

class BiseNetV2:
    def __init__(self, cfg):
        # set model hyper params
        self._class_nums = cfg.DATASET.NUM_CLASSES
        self._weights_decay = cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = cfg.SOLVER.LOSS_TYPE
        self._enable_ohem = cfg.SOLVER.ENABLE_OHEM
        if self._enable_ohem:
            self._ohem_score_thresh = cfg.SOLVER.OHEM_SCORE_THRESH
            self._ohem_min_sample_nums = cfg.SOLVER.OHEM_MIN_SAMPLE_NUMS
        self._ge_expand_ratio = cfg.MODEL.GE_EXPAND_RATIO
        self._semantic_channel_ratio = cfg.MODEL.SEMANTIC_CHANNEL_LAMBDA
        self._seg_head_ratio = cfg.MODEL.SEGHEAD_CHANNEL_EXPAND_RATIO

        self.input_size = [1080, 1920, 3]
        self.model = self.build_net()

    def build_detail_branch(self, input_tensor):
        result = input_tensor
        """
        params = [
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        """

        # stage 1
        ###########################################
        ksize = 3
        output_channels = 64
        strides = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=True)
        ksize = 3
        output_channels = 64
        strides = 1
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=True)

        # stage 2
        ###########################################
        ksize = 3
        output_channels = 64
        strides = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=True)
        ksize = 3
        output_channels = 64
        strides = 1
        repeat_times = 2
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=True)


        # stage 3
        ###########################################
        ksize = 3
        output_channels = 128
        strides = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=False)
        ksize = 3
        output_channels = 128
        strides = 1
        repeat_times = 2
        for repeat_index in range(repeat_times):
            result = conv_block(input_tensor=result, k_size=ksize, output_channels=output_channels, stride=strides,
                                padding='SAME', use_bias=False, need_activate=True)
        return result

    def build_semantic_branch(self, input_tensor, prepare_data_for_booster=False):
        result = input_tensor
        seg_head_inputs = {}
        source_input_tensor_size = input_tensor.get_shape().as_list()[1:3]

        """
        params = [
            ('stage_1', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 1)]),
            ('stage_2', [('conv_block', 3, 64, 2, 1), ('conv_block', 3, 64, 1, 2)]),
            ('stage_3', [('conv_block', 3, 128, 2, 1), ('conv_block', 3, 128, 1, 2)]),
        ]
        """

        stage_1_channels = int(64 * self._semantic_channel_ratio)
        stage_3_channels = int(128 * self._semantic_channel_ratio)
        params = [
            ('stage_1', [('se', 3, stage_1_channels, 1, 4, 1)]),
            ('stage_3', [('ge', 3, stage_3_channels, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels, self._ge_expand_ratio, 1, 1)]),
            ('stage_4', [('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 2, self._ge_expand_ratio, 1, 1)]),
            ('stage_5', [('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 2, 1),
                         ('ge', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 3),
                         ('ce', 3, stage_3_channels * 4, self._ge_expand_ratio, 1, 1)])
        ]

        # stage 1
        ##################################################
        seg_head_input = input_tensor
        output_channels = stage_1_channels
        expand_ratio = 1
        stride = 4
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _StemBlock(input_tensor=result, output_channels=output_channels)
            seg_head_input = result

        if prepare_data_for_booster:
            result_tensor_size = result.get_shape().as_list()[1:3]
            result_tensor_dims = result.get_shape().as_list()[-1]
            upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
            feature_dims = result_tensor_dims * self._seg_head_ratio
            seg_head_inputs['stage_1'] = _SegmentationHead(
                input_tensor=seg_head_input,
                upsample_ratio=upsample_ratio,
                feature_dims=feature_dims,
                classes_nums=self._class_nums
            )

        # stage 3
        ##################################################
        seg_head_input = input_tensor
        output_channels = stage_3_channels
        expand_ratio = self._ge_expand_ratio
        stride = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result
        seg_head_input = input_tensor
        output_channels = stage_3_channels
        expand_ratio = self._ge_expand_ratio
        stride = 1
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result

        if prepare_data_for_booster:
            result_tensor_size = result.get_shape().as_list()[1:3]
            result_tensor_dims = result.get_shape().as_list()[-1]
            upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
            feature_dims = result_tensor_dims * self._seg_head_ratio
            seg_head_inputs['stage_3'] = _SegmentationHead(
                input_tensor=seg_head_input,
                upsample_ratio=upsample_ratio,
                feature_dims=feature_dims,
                classes_nums=self._class_nums
            )

        # stage 4
        ##################################################
        seg_head_input = input_tensor
        output_channels = stage_3_channels * 2
        expand_ratio = self._ge_expand_ratio
        stride = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result
        seg_head_input = input_tensor
        output_channels = stage_3_channels * 2
        expand_ratio = self._ge_expand_ratio
        stride = 1
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result

        if prepare_data_for_booster:
            result_tensor_size = result.get_shape().as_list()[1:3]
            result_tensor_dims = result.get_shape().as_list()[-1]
            upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
            feature_dims = result_tensor_dims * self._seg_head_ratio
            seg_head_inputs['stage_4'] = _SegmentationHead(
                input_tensor=seg_head_input,
                upsample_ratio=upsample_ratio,
                feature_dims=feature_dims,
                classes_nums=self._class_nums
            )

        # stage 5
        ##################################################
        seg_head_input = input_tensor
        output_channels = stage_3_channels * 4
        expand_ratio = self._ge_expand_ratio
        stride = 2
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result
        seg_head_input = input_tensor
        output_channels = stage_3_channels * 4
        expand_ratio = self._ge_expand_ratio
        stride = 1
        repeat_times = 3
        for repeat_index in range(repeat_times):
            result = _GatherExpansion(input_tensor=result, output_channels=output_channels, stride=stride, e=expand_ratio)
            seg_head_input = result

        seg_head_input = input_tensor
        output_channels = stage_3_channels * 4
        expand_ratio = self._ge_expand_ratio
        stride = 1
        repeat_times = 1
        for repeat_index in range(repeat_times):
            result = _ContextEmbedding(input_tensor=result)

        if prepare_data_for_booster:
            result_tensor_size = result.get_shape().as_list()[1:3]
            result_tensor_dims = result.get_shape().as_list()[-1]
            upsample_ratio = int(source_input_tensor_size[0] / result_tensor_size[0])
            feature_dims = result_tensor_dims * self._seg_head_ratio
            seg_head_inputs['stage_5'] = _SegmentationHead(
                input_tensor=seg_head_input,
                upsample_ratio=upsample_ratio,
                feature_dims=feature_dims,
                classes_nums=self._class_nums
            )

        return result, seg_head_inputs

    def build_aggregation_branch(self, detail_output, semantic_output):
        result = _GuidedAggregation(detail_input_tensor=detail_output, semantic_input_tensor=semantic_output)
        return result

    @classmethod
    def _compute_cross_entropy_loss(cls, seg_logits, labels, class_nums):
        """

        :param seg_logits:
        :param labels:
        :param class_nums:
        :param name:
        :return:
        """

        # first check if the logits' shape is matched with the labels'
        seg_logits_shape = seg_logits.shape[1:3]
        labels_shape = labels.shape[1:3]
        seg_logits = tf.cond(
            tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
            true_fn=lambda: seg_logits,
            false_fn=lambda: tf.image.resize(seg_logits, labels_shape)
        )
        seg_logits = tf.reshape(seg_logits, [-1, class_nums])
        labels = tf.reshape(labels, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(labels, class_nums - 1)), 1)
        seg_logits = tf.gather(seg_logits, indices)
        labels = tf.cast(tf.gather(labels, indices), tf.int32)

        # compute cross entropy loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=seg_logits
            ),
            name='cross_entropy_loss'
        )
        return loss

    @classmethod
    def _compute_ohem_cross_entropy_loss(cls, seg_logits, labels, class_nums, thresh, n_min):
        # first check if the logits' shape is matched with the labels'
        seg_logits_shape = seg_logits.shape[1:3]
        labels_shape = labels.shape[1:3]
        seg_logits = tf.cond(
            tf.reduce_all(tf.equal(seg_logits_shape, labels_shape)),
            true_fn=lambda: seg_logits,
            false_fn=lambda: tf.image.resize(seg_logits, labels_shape)
        )
        seg_logits = tf.reshape(seg_logits, [-1, class_nums])
        labels = tf.reshape(labels, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(labels, class_nums - 1)), 1)
        seg_logits = tf.gather(seg_logits, indices)
        labels = tf.cast(tf.gather(labels, indices), tf.int32)

        # compute cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=seg_logits
        )
        loss, _ = tf.nn.top_k(loss, tf.size(loss), sorted=True)
        # apply ohem
        ohem_thresh = tf.multiply(-1.0, tf.math.log(thresh))
        ohem_cond = tf.greater(loss[n_min], ohem_thresh)
        loss_select = tf.cond(
            pred=ohem_cond,
            true_fn=lambda: tf.gather(loss, tf.squeeze(tf.where(tf.greater(loss, ohem_thresh)), 1)),
            false_fn=lambda: loss[:n_min]
        )
        loss_value = tf.reduce_mean(loss_select, name='ohem_cross_entropy_loss')

        return loss_value

    @classmethod
    def _compute_dice_loss(cls, seg_logits, labels, class_nums):
        """
        dice loss is combined with bce loss here
        :param seg_logits:
        :param labels:
        :param class_nums:
        :param name:
        :return:
        """
        def __dice_loss(_y_pred, _y_true):
            """

            :param _y_pred:
            :param _y_true:
            :return:
            """
            _intersection = tf.reduce_sum(_y_true * _y_pred, axis=-1)
            _l = tf.reduce_sum(_y_pred * _y_pred, axis=-1)
            _r = tf.reduce_sum(_y_true * _y_true, axis=-1)
            _dice = (2.0 * _intersection + 1e-5) / (_l + _r + 1e-5)
            _dice = tf.reduce_mean(_dice)
            return 1.0 - _dice


        # compute dice loss
        local_label_tensor = tf.one_hot(labels, depth=class_nums, dtype=tf.float32)
        principal_loss_dice = __dice_loss(tf.nn.softmax(seg_logits), local_label_tensor)
        principal_loss_dice = tf.identity(principal_loss_dice, name='principal_loss_dice')

        # compute bce loss
        principal_loss_bce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=seg_logits)
        )
        principal_loss_bce = tf.identity(principal_loss_bce, name='principal_loss_bce')

        total_loss = principal_loss_dice + principal_loss_bce
        total_loss = tf.identity(total_loss, name='dice_loss')
        return total_loss

    @classmethod
    def _compute_l2_reg_loss(cls, var_list, weights_decay):
        """

        :param var_list:
        :param weights_decay:
        :param name:
        :return:
        """

        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in var_list:
            if 'beta' in vv.name or 'gamma' in vv.name or 'b:0' in vv.name.split('/')[-1]:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')

        return l2_reg_loss



    def build_net(self):
        input_layer = tf.keras.layers.Input(self.input_size)

        self.bisenetv2_detail_branch_output = self.build_detail_branch(input_tensor=input_layer)
        self.bisenetv2_semantic_branch_output, self.segment_head_inputs = self.build_semantic_branch(input_tensor=input_layer)
        self.bisenetv2_aggregation_output = self.build_aggregation_branch(
            detail_output=self.bisenetv2_detail_branch_output,
            semantic_output=self.bisenetv2_semantic_branch_output
        )

        output_tensors = []

        # build segmentation head
        segment_logits = _SegmentationHead(
            input_tensor=self.bisenetv2_aggregation_output,
            upsample_ratio=8,
            feature_dims=self._seg_head_ratio * self.bisenetv2_aggregation_output.get_shape().as_list()[-1],
            classes_nums=self._class_nums
        )
        self.segment_head_inputs['seg_head'] = segment_logits

        output_tensors.append(self.bisenetv2_aggregation_output)
        output_tensors.append(self.segment_head_inputs)

        return tf.keras.Model(input_layer, output_tensors)

    def compute_loss(self, input_tensor, label_tensor, reuse=False, generate_segmentation=False):
        aggregation_output, semantic_branch_seg_logits = self.model(input_tensor, training=True)

        segment_loss = tf.constant(0.0, tf.float32)

        for stage_name, seg_logits in semantic_branch_seg_logits.items():
            loss_stage_name = '{:s}_segmentation_loss'.format(stage_name)
            if self._loss_type == 'cross_entropy':
                if not self._enable_ohem:
                    segment_loss += self._compute_cross_entropy_loss(
                        seg_logits=seg_logits,
                        labels=label_tensor,
                        class_nums=self._class_nums
                    )
                else:
                    segment_loss += self._compute_ohem_cross_entropy_loss(
                        seg_logits=seg_logits,
                        labels=label_tensor,
                        class_nums=self._class_nums,
                        thresh=self._ohem_score_thresh,
                        n_min=self._ohem_min_sample_nums
                    )
            elif self._loss_type == 'dice':
                segment_loss += self._compute_dice_loss(
                    seg_logits=seg_logits,
                    labels=label_tensor,
                    class_nums=self._class_nums,
                )
            else:
                raise NotImplementedError('Not supported loss of type: {:s}'.format(self._loss_type))

        #l2_reg_loss = self._compute_l2_reg_loss(
        #    var_list=self.model.trainalbe_variables,
        #    weights_decay=self._weights_decay
        #)
        l2_reg_loss = 0
        total_loss = segment_loss + l2_reg_loss
        total_loss = tf.identity(total_loss, name='total_loss')


        ret = {
            'total_loss': total_loss,
            'l2_loss': l2_reg_loss,
            'segment_loss': segment_loss
        }
        if generate_segmentation:
            segment_logits = semantic_branch_seg_logits['seg_head']
            segment_score = tf.nn.softmax(logits=segment_logits)
            segment_prediction = tf.argmax(segment_score, axis=-1)

            ret['segment_prediction'] = segment_prediction
        return ret


    def inference(self, input_tensor):
        aggregation_output, semantic_branch_seg_logits = self.model(input_tensor, training=False)

        segment_logits = semantic_branch_seg_logits['seg_head']
        segment_score = tf.nn.softmax(logits=segment_logits)
        segment_prediction = tf.argmax(segment_score, axis=-1)
        return segment_prediction

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False, **kwargs):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def maxpooling(input_tensor, kernel_size, stride, padding='SAME'):
    res = tf.keras.layers.MaxPool2D(kernel_size, strides=stride,padding=padding)(input_tensor)
    return res


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


class BiseNetV2:
    def __init__(self):
        self._class_nums = 2
        self._weights_decay = 0.0005
        self._loss_type = 'cross_entropy'
        self._enable_ohem = True
        if self._enable_ohem:
            self._ohem_score_thresh = 0.65
            self._ohem_min_sample_nums = 262144
        self._ge_expand_ratio = 6
        self._semantic_channel_ratio = 0.25
        self._seg_head_ratio = 2

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

    def build_net(self):
        input_layer = tf.keras.layers.Input(self.input_size)

        self.bisenetv2_detail_branch_output = self.build_detail_branch(input_tensor=input_layer)
        self.bisenetv2_semantic_branch_output, self.segment_head_inputs = self.build_semantic_branch(input_tensor=input_layer)
        self.bisenetv2_aggregation_output = self.build_aggregation_branch(
            detail_output=self.bisenetv2_detail_branch_output,
            semantic_output=self.bisenetv2_semantic_branch_output
        )

        output_tensors = []

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


def bisenet_v2():
    return BiseNetV2().build_net()

# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:shufflenet.py
# software: PyCharm


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class ConvBNRelu(keras.Model):

    def __init__(self, channels, kernel_size, strides):
        super(ConvBNRelu, self).__init__()
        self.conv = layers.Conv2D(channels, kernel_size, strides, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def __call__(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.relu(x)

        return x


class DepthwiseConvBNRelu(keras.Model):

    def __init__(self, kernel_size, strides):
        super(DepthwiseConvBNRelu, self).__init__()
        self.depth_wise = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()

    def __call__(self, inputs, training=True):
        x = self.depth_wise(inputs)
        x = self.bn(x, training)

        return x


class ChannelShuffle(keras.Model):

    def __init__(self, group):
        super(ChannelShuffle, self).__init__()
        self.group = group

    def __call__(self, inputs):
        # inputs [batch, h, w, channel]
        shape = inputs.shape
        # batch = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        # assert c % self.group == 0, 'c % group needs to be zero!'

        inputs = tf.reshape(inputs, shape=[-1, h, w, c // self.group, self.group])
        inputs = tf.transpose(inputs, [0, 1, 2, 4, 3])
        inputs = tf.reshape(inputs, shape=(-1, h, w, c))

        return inputs


class ShuffleBlock(keras.Model):

    def __init__(self, channels, strides, split_ratio=0.5):
        super(ShuffleBlock, self).__init__()
        self.split_ratio = split_ratio
        self.conv1 = ConvBNRelu(channels // 2, 1, 1)
        self.depth_wise = DepthwiseConvBNRelu(3, strides=strides)
        self.conv2 = ConvBNRelu(channels // 2, 1, 1)
        self.shuffle = ChannelShuffle(group=2)

    def __call__(self, inputs, training):
        # 1.channels split
        x1, x2 = tf.split(inputs, num_or_size_splits=int(1 / self.split_ratio), axis=-1)
        # 2.conv_1X1, depthwise_3X3, conv_1X1
        x2 = self.conv1(x2, training)
        x2 = self.depth_wise(x2, training)
        x2 = self.conv2(x2, training)
        # 3.concatenate x1 and x2 to make information communicate
        feature = layers.Concatenate()([x1, x2])
        # 4.channel shuffle
        res = self.shuffle(feature)

        return res


class ShuffleConvBlock(keras.Model):

    def __init__(self, in_channels, out_channels, strides):
        super(ShuffleConvBlock, self).__init__()
        self.conv1 = ConvBNRelu(out_channels - in_channels, 1, 1)
        self.depth_wise = DepthwiseConvBNRelu(3, strides=strides)
        self.conv2 = ConvBNRelu(out_channels - in_channels, 1, 1)
        self.depth_wise_lateral = DepthwiseConvBNRelu(3, strides=strides)
        self.conv_lateral = ConvBNRelu(in_channels, 1, 1)
        self.shuffle = ChannelShuffle(group=2)

    def __call__(self, inputs, training):
        x1, x2 = inputs, inputs
        x2 = self.conv1(x2, training)
        x2 = self.depth_wise(x2, training)
        x2 = self.conv2(x2, training)

        x1 = self.depth_wise_lateral(x1, training)
        x1 = self.conv_lateral(x1, training)

        feature = layers.Concatenate()([x1, x2])
        res = self.shuffle(feature)

        return res


class ShuffleNetV2(keras.Model):
    """ShuffleNetV2
    How to reduce MAC:
    1.make channels_in == channels_out
    2.don't use group convolution
    3.change add to concatenate
    4.don't make model fragmented
    So, use:
    1.conv_1X1
    2.depthwise and pointwise
    3.concatenate rather than add
    4.maybe shuffle block can promote accuracy

    """

    def __init__(self, channels=[24, 116, 232, 464, 1024]):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = layers.Conv2D(channels[0], 3, 2, padding='same')
        self.pool = layers.MaxPool2D(3, strides=2, padding='same')
        self.stage1 = ShuffleNetStage(repeat=3, in_channels=channels[0], out_channels=channels[1])
        self.stage2 = ShuffleNetStage(repeat=7, in_channels=channels[1], out_channels=channels[2])
        self.stage3 = ShuffleNetStage(repeat=3, in_channels=channels[2], out_channels=channels[3])
        self.conv2 = layers.Conv2D(channels[4], kernel_size=1, padding='same')

    def __call__(self, inputs, training):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.stage1(x, training)
        x = self.stage2(x, training)
        x = self.stage3(x, training)
        x = self.conv2(x)

        return x


class ShuffleNetStage(keras.Model):

    def __init__(self, repeat, in_channels, out_channels):
        super(ShuffleNetStage, self).__init__()
        self.shuffle_conv_block = ShuffleConvBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   strides=2)
        self.convs = []
        for i in range(repeat):
            self.convs.append(ShuffleBlock(channels=out_channels,
                                           strides=1))

    def __call__(self, inputs, training):
        x = self.shuffle_conv_block(inputs, training)
        for conv in self.convs:
            x = conv(x, training)

        return x


if __name__ == '__main__':
    shufflenet_v2 = ShuffleNetV2()
    inputs_ = keras.Input(shape=(224, 224, 3))
    res = shufflenet_v2(inputs_, training=True)
    model = keras.Model(inputs_, res)
    model.summary()

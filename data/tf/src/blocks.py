import tensorflow as tf
from tensorflow.keras import layers


def conv3x3(kernels, strides=1):
    return layers.Conv2D(kernels, 3, strides=strides, padding='same', use_bias=False)


def conv1x1(kernels, strides=1):
    return layers.Conv2D(kernels, 1, strides=strides, use_bias=False)


class BasicBlock(layers.Layer):

    expansion = 1

    def __init__(self,
                 kernels,
                 strides=1,
                 downsample=None,
                 norm_layer=None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer()

        self.conv1 = conv3x3(kernels, strides)
        self.bn1 = norm_layer()
        self.relu = layers.ReLU()
        self.conv2 = conv3x3(kernels)

        if start_block or end_block:
            self.bn2 = norm_layer()

        self.downsample = downsample
        self.strides = strides

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x, training=False):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x, training=training)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn2(out, training=training)
            out = self.relu(out)

        return out


class Bottleneck(layers.Layer):

    expansion = 4

    def __init__(self,
                 kernels,
                 strides=1,
                 downsample=None,
                 norm_layer=None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer()

        self.conv1 = conv1x1(kernels)
        self.bn1 = norm_layer()
        self.conv2 = conv3x3(kernels, strides)
        self.bn2 = norm_layer()
        self.conv3 = conv1x1(kernels * self.expansion)

        if start_block or end_block:
            self.bn3 = norm_layer()

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.strides = strides

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x, training=False):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out

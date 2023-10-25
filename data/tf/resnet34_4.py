import tensorflow as tf
from keras import Sequential, regularizers


def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=regularizers.l2(5e-4))


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=32):
        super(ChannelAttention, self).__init__()
        self.avg = tf.keras.layers.GlobalAveragePooling2D()
        self.max = tf.keras.layers.GlobalMaxPooling2D()
        self.conv1 = tf.keras.layers.Conv2D(in_planes // ratio, kernel_size=1, strides=1, padding='same',
                                            kernel_regularizer=regularizers.l2(5e-4),
                                            use_bias=True, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                            kernel_regularizer=regularizers.l2(5e-4),
                                            use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = tf.keras.layers.Reshape((1, 1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        max = tf.keras.layers.Reshape((1, 1, max.shape[1]))(max)  # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)  # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = regularized_padded_conv(out_channels, kernel_size=3, strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = regularized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels,
                                                                kernel_size=1, strides=stride),
                                        tf.keras.layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.ca(out) * out
        out = self.sa(out) * out

        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = regularized_padded_conv(out_channels, 1, 1)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = regularized_padded_conv(out_channels, 3, strides)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = regularized_padded_conv(out_channels * self.expansion, 1, 1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.ca = ChannelAttention(self.expansion * out_channels)
        self.sa = SpatialAttention()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels, kernel_size=1,
                                                                strides=strides),
                                        tf.keras.layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out = self.ca(out) * out
        out = self.sa(out) * out

        out = out + self.shortcut(x, training)
        out = tf.nn.relu(out)

        return out


class ResNet(tf.keras.Model):

    def __init__(self, blocks, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.stem = Sequential([regularized_padded_conv(64, kernel_size=3, strides=1),
                                tf.keras.layers.BatchNormalization()])

        self.layer1 = self.build_resblock(blocks, 64, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(blocks, 128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(blocks, 256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(blocks, 512, layer_dims[3], stride=2)

    def build_resblock(self, blocks, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [1]*3 = [1, 1, 1]
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(blocks(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks

    def call(self, inputs, training=False):
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        return out


def resnet34_4():
    return ResNet(BasicBlock, [3, 4, 6, 3])

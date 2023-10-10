# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:squeezenet.py
# software: PyCharm


import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf


class SqueezeNet(keras.Model):

    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act1 = Mish()
        self.pool1 = layers.MaxPool2D(3, strides=2, padding='same')
        self.fire_block2 = FireBlock(16, 64, 64)
        self.fire_block3 = FireBlock(16, 64, 64)
        self.fire_block4 = FireBlock(32, 128, 128)
        self.pool4 = layers.MaxPool2D(3, strides=2, padding='same')
        self.fire_block5 = FireBlock(32, 128, 128)
        self.fire_block6 = FireBlock(48, 192, 192)
        self.fire_block7 = FireBlock(48, 192, 192)
        self.fire_block8 = FireBlock(64, 256, 256)
        self.pool8 = layers.MaxPool2D(3, strides=2, padding='same')
        self.fire_block9 = FireBlock(64, 256, 256)

    def __call__(self, inputs=None):
        if not inputs:
            inputs = keras.Input(shape=(224, 224, 3))

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)  # C2
        x = self.fire_block2(x)
        x = self.fire_block3(x)
        x = self.fire_block4(x)
        x = self.pool4(x)  # C3
        x = self.fire_block5(x)
        x = self.fire_block6(x)
        x = self.fire_block7(x)
        x = self.fire_block8(x)
        x = self.pool8(x)  # C4
        x = self.fire_block9(x)
        squeezenet = keras.Model(inputs, x)
        return squeezenet


class FireBlock(layers.Layer):

    def __init__(self, filter1, filter2, filter3):
        super(FireBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=filter1, kernel_size=1, padding='same', use_bias=False)
        self.conv2 = layers.Conv2D(filters=filter2, kernel_size=1, padding='same', use_bias=False)
        self.conv3 = layers.Conv2D(filters=filter3, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

    def __call__(self, inputs):
        squeeze_x = self.conv1(inputs)
        squeeze_x = self.bn1(squeeze_x)
        squeeze_x = Mish()(squeeze_x)
        expand_x1 = self.conv2(squeeze_x)
        expand_x1 = self.bn2(expand_x1)
        expand_x1 = Mish()(expand_x1)
        expand_x3 = self.conv3(squeeze_x)
        expand_x3 = self.bn3(expand_x3)
        expand_x3 = Mish()(expand_x3)

        merge_x = layers.Concatenate()([expand_x1, expand_x3])

        return merge_x


class Mish(layers.Layer):

    def __init__(self):
        super(Mish, self).__init__()

    def __call__(self, inputs):
        return tf.multiply(inputs, tf.tanh(tf.nn.softplus(inputs)))


if __name__ == '__main__':
    squeeze_net = SqueezeNet()
    model = squeeze_net()
    model.summary()

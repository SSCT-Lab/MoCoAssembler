# -*- coding: utf-8 -*-

"""
@Title   :  TensorFlow implementation of GoogLeNet
@Time    :  Mar. 11th, 2023
@Author  :  Biophilia Wu
@Email   :  BiophiliaSWDA@163.com
"""

import tensorflow as tf
from tensorflow.python.keras import layers, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout, Softmax


def GoogLeNet(class_num=1000, input_shape=(224, 224, 3), aux_logits=True):
    input_tensor = Input(shape=input_shape, dtype="float32")

    x = Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu", name="conv2d_1")(input_tensor)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool_1")(x)

    x = Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu", name="conv2d_2")(x)
    x = Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu", name="conv2d_3")(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool_2")(x)

    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool_3")(x)

    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
    if aux_logits:
        aux1 = InceptionAux(class_num, name="aux_1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    if aux_logits:
        aux2 = InceptionAux(class_num, name="aux_2")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool_4")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    x = AveragePooling2D(pool_size=7, strides=1, name="avgpool_1")(x)

    x = Flatten()(x)

    x = Dropout(rate=0.4)(x)
    output_tensor = Dense(class_num, activation="softmax", name="output")(x)

    if aux_logits:
        model = Model(inputs=input_tensor, outputs=[aux1, aux2, output_tensor])
    else:
        model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = Conv2D(ch1x1, kernel_size=1, activation="relu")

        self.branch2 = Sequential([
            Conv2D(ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu"),
            Conv2D(ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")])

        self.branch3 = Sequential([
            Conv2D(ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu"),
            Conv2D(ch5x5, kernel_size=5, strides=1, padding="same", activation="relu")])

        self.branch4 = Sequential([
            MaxPooling2D(pool_size=3, strides=1, padding="same"),
            Conv2D(pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(layers.Layer):
    def __init__(self, class_num, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = AveragePooling2D(pool_size=5, strides=3)
        self.conv = Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu")

        self.fc1 = Dense(1024, activation="relu")
        self.fc2 = Dense(class_num)
        self.flatten = Flatten()
        self.softmax = Softmax()

    def call(self, inputs, **kwargs):
        x = self.averagePool(inputs)
        x = self.conv(x)
        x = self.flatten(x)

        x = Dropout(rate=0.7)(x)
        x = self.fc1(x)

        x = Dropout(rate=0.7)(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = GoogLeNet(class_num=1000, input_shape=(224, 224, 3))
    net.summary()

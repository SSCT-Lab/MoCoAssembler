# -*- coding: utf-8 -*-

"""
@Title   :  TensorFlow implementation of InceptionV3
@Time    :  Mar. 12th, 2023
@Author  :  Biophilia Wu
@Email   :  BiophiliaSWDA@163.com
"""

import tensorflow as tf
from tensorflow.python.keras import layers, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Softmax
from collections import namedtuple
_InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])


class InceptionAux(layers.Layer):
    def __init__(self, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = AveragePooling2D(pool_size=(5, 5), strides=3, padding="same")
        self.conv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.conv2 = Conv2D(filters=768, kernel_size=(5, 5), strides=1, padding="same", activation="relu")
        self.globalAveragePool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(units=num_classes, activation=tf.keras.activations.linear)

    def call(self, inputs, **kwargs):
        x = self.averagePool(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.globalAveragePool(x)
        x = self.flat(x)
        x = self.fc(x)

        return x


class InceptionA(layers.Layer):
    def __init__(self, filter_num):
        super(InceptionA, self).__init__()

        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")

        self.branch2 = Sequential([
            Conv2D(filters=48, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding="same", activation="relu")]
        )

        self.branch3 = Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu"),
            Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu")]
        )

        self.branch4 = Sequential([
            AveragePooling2D(pool_size=(3, 3), strides=1, padding="same"),
            Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu")]
        )

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionB(layers.Layer):
    def __init__(self):
        super(InceptionB, self).__init__()

        self.branch1 = Conv2D(filters=384, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")

        self.branch2 = Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu"),
            Conv2D(filters=96, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")
        ])

        self.branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3])
        return outputs


class InceptionC(layers.Layer):
    def __init__(self, filter_num):
        super(InceptionC, self).__init__()

        self.branch1 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")

        self.branch2 = Sequential([
            Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=filter_num, kernel_size=(1, 7), strides=1, padding="same", activation="relu"),
            Conv2D(filters=192, kernel_size=(7, 1), strides=1, padding="same", activation="relu")
        ])

        self.branch3 = Sequential([
            Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=filter_num, kernel_size=(7, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=filter_num, kernel_size=(1, 7), strides=1, padding="same", activation="relu"),
            Conv2D(filters=filter_num, kernel_size=(7, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=192, kernel_size=(1, 7), strides=1, padding="same", activation="relu")
        ])

        self.branch4 = Sequential([
            MaxPooling2D(pool_size=(3, 3), strides=1, padding="same"),
            Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        ])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionD(layers.Layer):
    def __init__(self):
        super(InceptionD, self).__init__()

        self.branch1 = Sequential([
            Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=320, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")
        ])

        self.branch2 = Sequential([
            Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=192, kernel_size=(1, 7), strides=1, padding="same", activation="relu"),
            Conv2D(filters=192, kernel_size=(7, 1), strides=1, padding="same", activation="relu"),
            Conv2D(filters=192, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")
        ])

        self.branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3])
        return outputs


class InceptionE(layers.Layer):
    def __init__(self):
        super(InceptionE, self).__init__()
        self.conv1 = Conv2D(filters=320, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.conv2 = Conv2D(filters=384, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.conv3 = Conv2D(filters=448, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.conv4 = Conv2D(filters=384, kernel_size=(1, 3), strides=1, padding="same", activation="relu")
        self.conv5 = Conv2D(filters=384, kernel_size=(3, 1), strides=1, padding="same", activation="relu")
        self.conv6 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        self.conv7 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.avgpool = AveragePooling2D(pool_size=(3, 3), strides=1, padding="same")

    def call(self, inputs, **kwargs):
        branch1 = self.conv1(inputs)

        branch2 = self.conv2(inputs)
        branch2a = self.conv4(branch2)
        branch2b = self.conv5(branch2)
        branch2 = layers.concatenate([branch2a, branch2b], axis=-1)

        branch3 = self.conv3(inputs)
        branch3 = self.conv6(branch3)
        branch3a = self.conv4(branch3)
        branch3b = self.conv5(branch3)
        branch3 = layers.concatenate([branch3a, branch3b], axis=-1)

        branch4 = self.avgpool(inputs)
        branch4 = self.conv7(branch4)

        outputs = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
        return outputs


def InceptionV3(class_num=1000, input_shape=(299, 299, 3), aux_logits=True):
    input_tensor = Input(shape=input_shape, dtype="float32")

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(input_tensor)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid", activation="relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    x = Conv2D(filters=80, kernel_size=(1, 1), strides=1, padding="valid", activation="relu")(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    # 3 InceptionA
    x = InceptionA(filter_num=32)(x)
    x = InceptionA(filter_num=64)(x)
    x = InceptionA(filter_num=64)(x)

    # 1 InceptionB
    x = InceptionB()(x)

    # 4 InceptionC
    x = InceptionC(filter_num=128)(x)
    x = InceptionC(filter_num=160)(x)
    x = InceptionC(filter_num=160)(x)
    x = InceptionC(filter_num=192)(x)

    # 1 InceptionAux
    if aux_logits:
        aux = InceptionAux(num_classes=class_num)(x)

    # 1 InceptionD
    x = InceptionD()(x)

    # 2 InceptionE
    x = InceptionE()(x)
    x = InceptionE()(x)

    x = AveragePooling2D(pool_size=(8, 8), strides=1, padding="valid")(x)
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=class_num, activation=tf.keras.activations.linear)(x)

    if aux_logits:
        model = Model(inputs=input_tensor, outputs=[aux, x])
    else:
        model = Model(inputs=input_tensor, outputs=x)
    return model


if __name__ == '__main__':
    net = InceptionV3()

    net.build(input_shape=(None, 299, 299, 3))
    net.summary()



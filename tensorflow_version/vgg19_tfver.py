# -*- coding: utf-8 -*-

"""
@Title   :  TensorFlow implementation of VGG19
@Time    :  Mar. 11th, 2023
@Author  :  Biophilia Wu
@Email   :  BiophiliaSWDA@163.com
"""

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def VGG19(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = Input(shape=input_shape, dtype="float32")
    # 1st block
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1a")(input_tensor)
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1b")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)

    # 2nd block
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same", name="conv2a")(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same", name="conv2b")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)

    # 3rd block
    x = Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3a")(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3b")(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3c")(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3d")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)

    # 4th block
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4a")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4b")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4c")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4d")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="pool4")(x)

    # 5th block
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5a")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5b")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5c")(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5d")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="pool5")(x)

    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation="relu",  name="fc6")(x)

    x = Dense(4096, activation="relu", name="fc7")(x)

    output_tensor = Dense(class_num, activation="softmax", name="fc8")(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    net = VGG19(class_num=1000, input_shape=(224, 224, 3))
    net.summary()

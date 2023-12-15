import numpy as np
import tensorflow as tf
from tensorflow import keras
from moco_tf.config.paths import DATASETS_PATH

def pointnet(input_shape):
    # input layers
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
    x = input_tensor

    # hidden layers
    x = keras.layers.Conv1D(kernel_size=1, filters=64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv1D(kernel_size=1, filters=128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv1D(kernel_size=1, filters=1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=512)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(units=256)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(units=10)(x)

    # output layers
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=10, activation='softmax')(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    model = pointnet(input_shape=(5, 3))
    x = tf.random.normal(shape=(1,) + (5, 3))
    y = model(x)
    return model



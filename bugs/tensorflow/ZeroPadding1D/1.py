import numpy as np
import tensorflow as tf
from tensorflow import keras
from moco_tf.config.paths import DATASETS_PATH

def pointnet(input_shape):
    # input layers
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
    x = input_tensor

    # hidden layers
    x = keras.layers.Conv1D(kernel_size=1, filters=64, use_bias=True)(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.ReLU(threshold=0.24225966126127751)(x)
    x = keras.layers.Conv1DTranspose(kernel_size=1, filters=128)(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv1DTranspose(kernel_size=1, filters=1024)(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.ReLU(threshold=0.690673942585456)(x)
    x = keras.layers.Flatten(data_format='channels_last')(x)
    x = keras.layers.ZeroPadding1D(padding={'padding': 1})(x)

    # output layers
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=10, activation='softmax')(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    model = pointnet(input_shape=(5, 3))
    x = tf.random.normal(shape=(1,) + (5, 3))
    y = model(x)
    return model

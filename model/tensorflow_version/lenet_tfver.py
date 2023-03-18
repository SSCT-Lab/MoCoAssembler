from tensorflow import keras
import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, layers, optimizers, losses, metrics, Model

labels_num = 2 # 类别数

def lenet(label_num, input_shape):
    #网络层的搭建
    inputs = keras.Input(shape=input_shape)

    # 1st block
    x = layers.Conv2D(6, kernel_size=3, strides=1, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 2nd block
    x = layers.Conv2D(16, kernel_size=3, strides=1,activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 3rd block
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)


    outputs = layers.Dense(label_num)(x)

    model = Model(inputs, outputs)

    return model

if __name__ == '__main__':
    model = lenet(2, [32, 32, 1])
    model.summary()


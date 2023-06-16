import tensorflow as tf
from tensorflow import keras


def lenet(label_num=2, input_shape=(32, 32, 1)):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.Conv2D(filters=6, kernel_size=3, strides=1, activation='relu')(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 2nd block
    x = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 3rd block
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=120, activation='relu')(x)
    x = keras.layers.Dense(units=84, activation='relu')(x)
    x = keras.layers.Dense(units=label_num)(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


if __name__ == '__main__':
    model = lenet(2, [32, 32, 1])


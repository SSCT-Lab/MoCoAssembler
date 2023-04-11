import tensorflow as tf
from tensorflow.python.keras import Input, layers, models


def lenet(label_num, input_shape):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = layers.Conv2D(6, kernel_size=3, strides=1, activation='relu')(input_tensor)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 2nd block
    x = layers.Conv2D(16, kernel_size=3, strides=1, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 3rd block
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    x = layers.Dense(label_num)(x)

    output_tensor = x

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


if __name__ == '__main__':
    model = lenet(2, [32, 32, 1])
    model.summary()


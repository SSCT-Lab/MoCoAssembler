import tensorflow as tf
from tensorflow import keras


def InceptionV3(class_num=1000, input_shape=(299, 299, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(input_tensor)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid", activation="relu")(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    x = keras.layers.Conv2D(filters=80, kernel_size=(1, 1), strides=1, padding="valid", activation="relu")(x)
    x = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="valid", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    # 3 InceptionA
    x = InceptionA(x, filter_num=32)
    x = InceptionA(x, filter_num=64)
    x = InceptionA(x, filter_num=64)

    # 1 InceptionB
    x = InceptionB(x)

    # 4 InceptionC
    x = InceptionC(x, filter_num=128)
    x = InceptionC(x, filter_num=160)
    x = InceptionC(x, filter_num=160)
    x = InceptionC(x, filter_num=192)

    # 1 InceptionD
    x = InceptionD(x)

    # 2 InceptionE
    x = InceptionE(x)
    x = InceptionE(x)

    x = keras.layers.AveragePooling2D(pool_size=(8, 8), strides=1, padding="valid")(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=class_num, activation=tf.keras.activations.linear)(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def InceptionA(inputs, filter_num):
    x1 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding="same", activation="relu")(x2)

    x3 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x3 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x3)
    x3 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x3)

    x4 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding="same")(inputs)
    x4 = keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x4)

    outputs = keras.layers.concatenate([x1, x2, x3, x4])
    return outputs


def InceptionB(inputs):
    x1 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x2)
    x2 = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(x2)

    x3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(inputs)

    outputs = keras.layers.concatenate([x1, x2, x3])
    return outputs


def InceptionC(inputs, filter_num):
    x1 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 7), strides=1, padding="same", activation="relu")(x2)
    x2 = keras.layers.Conv2D(filters=192, kernel_size=(7, 1), strides=1, padding="same", activation="relu")(x2)

    x3 = keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x3 = keras.layers.Conv2D(filters=filter_num, kernel_size=(7, 1), strides=1, padding="same", activation="relu")(x3)
    x3 = keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 7), strides=1, padding="same", activation="relu")(x3)
    x3 = keras.layers.Conv2D(filters=filter_num, kernel_size=(7, 1), strides=1, padding="same", activation="relu")(x3)
    x3 = keras.layers.Conv2D(filters=192, kernel_size=(1, 7), strides=1, padding="same", activation="relu")(x3)

    x4 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(inputs)
    x4 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x4)

    outputs = keras.layers.concatenate([x1, x2, x3, x4])
    return outputs


def InceptionD(inputs):
    x1 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x1 = keras.layers.Conv2D(filters=320, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(x1)

    x2 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=192, kernel_size=(1, 7), strides=1, padding="same", activation="relu")(x2)
    x2 = keras.layers.Conv2D(filters=192, kernel_size=(7, 1), strides=1, padding="same", activation="relu")(x2)
    x2 = keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(x2)

    x3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(inputs)

    outputs = keras.layers.concatenate([x1, x2, x3])
    return outputs


def InceptionE(inputs):
    x1 = keras.layers.Conv2D(filters=320, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x2a = keras.layers.Conv2D(filters=384, kernel_size=(1, 3), strides=1, padding="same", activation="relu")(x2)
    x2b = keras.layers.Conv2D(filters=384, kernel_size=(3, 1), strides=1, padding="same", activation="relu")(x2)
    x2 = keras.layers.concatenate([x2a, x2b], axis=-1)

    x3 = keras.layers.Conv2D(filters=448, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(inputs)
    x3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x3)
    x3a = keras.layers.Conv2D(filters=384, kernel_size=(1, 3), strides=1, padding="same", activation="relu")(x3)
    x3b = keras.layers.Conv2D(filters=384, kernel_size=(3, 1), strides=1, padding="same", activation="relu")(x3)
    x3 = keras.layers.concatenate([x3a, x3b], axis=-1)

    x4 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding="same")(inputs)
    x4 = keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x4)

    outputs = keras.layers.concatenate([x1, x2, x3, x4], axis=-1)
    return outputs


if __name__ == '__main__':
    net = InceptionV3()

import tensorflow as tf
from tensorflow import keras


def xception(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    # 2nd block
    x = inceptionA(x, 128, 128, 128)

    # 3rd block
    x = inceptionA(x, 256, 256, 256)

    # 4th block
    x = inceptionA(x, 728, 728, 728)

    # 5th - 12th block
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)
    x = inceptionB(x)

    # 13th block
    x = inceptionA(x, 1024, 728, 1024)

    # 14th block
    x = keras.layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=class_num, activation="softmax")(x)
    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def inceptionA(inputs, filters1, filters2, filters3):
    residual = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False)(inputs)
    residual = keras.layers.BatchNormalization()(residual)

    x = keras.layers.SeparableConv2D(filters=filters2, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=filters3, kernel_size=(3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = keras.layers.add([x, residual])

    outputs = x
    return outputs


def inceptionB(inputs):
    residual = inputs
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.add([x, residual])
    outputs = x
    return outputs


if __name__ == "__main__":
    net = xception(class_num=3, input_shape=(224, 224, 3))
    net.summary()
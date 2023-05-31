import tensorflow as tf
from tensorflow import keras


def GoogLeNet(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu", name="keras.layers.conv2d_1")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="keras.layers.maxpool_1")(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu", name="keras.layers.conv2d_2")(x)
    x = keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu", name="keras.layers.conv2d_3")(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="keras.layers.maxpool_2")(x)

    x = inception(x, 64, 96, 128, 16, 32, 32)
    x = inception(x, 128, 128, 192, 32, 96, 64)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="keras.layers.maxpool_3")(x)

    x = inception(x, 192, 96, 208, 16, 48, 64)

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="keras.layers.maxpool_4")(x)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = inception(x, 384, 192, 384, 48, 128, 128)
    x = keras.layers.AveragePooling2D(pool_size=7, strides=1, name="avgpool_1")(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=class_num, activation="softmax", name="output")(x)
    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
    x1 = keras.layers.Conv2D(filters=ch1x1, kernel_size=1, activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")(x2)

    x3 = keras.layers.Conv2D(filters=ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x3 = keras.layers.Conv2D(filters=ch5x5, kernel_size=1, strides=1, padding="same", activation="relu")(x3)

    x4 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
    x4 = keras.layers.Conv2D(filters=pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")(x4)

    shape = tf.shape(x1)
    x2 = tf.reshape(x2, shape)
    x3 = tf.reshape(x3, shape)
    x4 = tf.reshape(x4, shape)
    outputs = keras.layers.concatenate([x1, x2, x3, x4])
    return outputs


if __name__ == '__main__':
    net = GoogLeNet(class_num=3, input_shape=(224, 224, 3))
    net.summary()

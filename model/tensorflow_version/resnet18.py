import tensorflow as tf
from tensorflow import keras


def resnet18(class_nums=3, input_shape=(28, 28, 1)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    # 1st block
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation="relu")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    # 2nd block
    x = inceptionB(x, filters=64, kernel_size=3, strides=1, padding='same')
    x = inceptionB(x, filters=64, kernel_size=3, strides=1, padding='same')

    # 3rd block
    x = inceptionA(x, filters=128, kernel_size=3, strides=2, padding='same')
    x = inceptionB(x, filters=128, kernel_size=3, strides=1, padding='same')

    # 4th block
    x = inceptionA(x, filters=256, kernel_size=3, strides=2, padding='same')
    x = inceptionB(x, filters=256, kernel_size=3, strides=1, padding='same')

    # 5th block
    x = inceptionA(x, filters=512, kernel_size=3, strides=2, padding='same')
    x = inceptionB(x, filters=512, kernel_size=3, strides=1, padding='same')

    # 6th block
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(units=class_nums, activation='softmax')(x)
    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, filters=64, kernel_size=3, strides=1, padding='same'):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu")(x)

    temp = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', activation="relu")(inputs)

    shape = tf.shape(x)
    temp = tf.reshape(temp, shape)
    outputs = keras.layers.add([x, temp])
    return outputs


def inceptionB(inputs, filters=64, kernel_size=3, strides=1, padding='same'):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu")(x)

    shape = tf.shape(x)
    inputs = tf.reshape(inputs, shape)

    outputs = keras.layers.add([x, inputs])
    return outputs


if __name__ == '__main__':
    model = resnet18(3, (28, 28,1))

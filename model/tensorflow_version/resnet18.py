import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def resnet18(class_num=1000, input_shape=(224, 224, 3)):
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

    output_tensor = keras.layers.Dense(units=class_num, activation="softmax")(keras.layers.Flatten()(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, filters=64, kernel_size=3, strides=1, padding='same'):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu")(x)

    temp = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', activation="relu")(inputs)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    temp = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(temp)
    outputs = keras.layers.add([x, temp])
    return outputs


def inceptionB(inputs, filters=64, kernel_size=3, strides=1, padding='same'):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu")(x)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    inputs = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(inputs)

    outputs = keras.layers.add([x, inputs])
    return outputs


def go():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=8192)])

    with tf.device("/GPU:0"):
        imagenet = np.load(DATASETS_PATH / "imagenet.npz")
        x_train = imagenet['x_test'][:100]
        y_train = imagenet['y_test'][:100]

        model = resnet18(1000, (224, 224, 3))
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=1)


if __name__ == "__main__":
    go()

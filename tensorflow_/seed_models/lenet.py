import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_.config.paths import DATASETS_PATH


def lenet(label_num=10, input_shape=(28, 28, 1)):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=200, activation="relu")(x)
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def go():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

    with tf.device("/GPU:0"):
        mnist = np.load(DATASETS_PATH / "mnist.npz")

    x_train = mnist['x_train'][:100]
    y_train = mnist['y_train'][:100]

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

    model = lenet(10, (28, 28, 1))

    model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()

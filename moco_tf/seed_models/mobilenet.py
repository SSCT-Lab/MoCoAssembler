import numpy as np
import tensorflow as tf
from tensorflow import keras
from moco_tf.config.paths import DATASETS_PATH


def mobilenet(label_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same")(input_tensor)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", groups=32)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=2, padding="same", groups=64)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", groups=128)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="valid",)(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", strides=2, padding="same", groups=128)(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", groups=256)(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", strides=2, padding="same", groups=256)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", strides=2, padding="same", groups=512)(x)
    x = keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", strides=2, padding="same", groups=1024)(x)
    x = keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), activation="relu", padding="valid")(x)

    x = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(x)

    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=8192)])

    with tf.device("/GPU:0"):
        cifar10 = np.load(DATASETS_PATH / "cifar10.npz")

    x_train = cifar10["x_train"][:100]
    y_train = cifar10["y_train"][:100]
    x_test = cifar10["x_test"][:50]
    y_test = cifar10["y_test"][:50]

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = mobilenet(10, (32, 32, 3))
    model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    model.evaluate(x_test, y_test, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()

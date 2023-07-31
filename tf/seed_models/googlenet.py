import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf.config.paths import DATASETS_PATH


def googlenet(num_class=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(x, 64, 96, 128, 16, 32, 32)
    x = inception(x, 128, 128, 192, 32, 96, 64)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(x, 192, 96, 208, 16, 48, 64)

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    output_tensor = keras.layers.Dense(units=num_class, activation="softmax")(keras.layers.Flatten()(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
    x1 = keras.layers.Conv2D(filters=ch1x1, kernel_size=1, activation="relu")(inputs)

    x2 = keras.layers.Conv2D(filters=ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(filters=ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")(x2)

    x3 = keras.layers.Conv2D(filters=ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x3 = keras.layers.Conv2D(filters=ch5x5, kernel_size=5, strides=1, padding="same", activation="relu")(x3)

    x4 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
    x4 = keras.layers.Conv2D(filters=pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")(x4)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x1)
    x2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x2)
    x3 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x3)
    x4 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x4)

    outputs = keras.layers.concatenate([x1, x2, x3, x4])
    return outputs


def go():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=8192)])

    with tf.device("/GPU:0"):
        imagenet = np.load(DATASETS_PATH / "imagenet.npz")
        print(imagenet.__dict__)
        x_train = imagenet["x_test"][:100]
        y_train = imagenet["y_test"][:100]

    model = googlenet(1000, (224, 224, 3))
    model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
    return model.count_params()


if __name__ == "__main__":
    go()

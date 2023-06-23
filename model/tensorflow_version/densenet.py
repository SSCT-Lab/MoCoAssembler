import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def densenet(num_class=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)
    features_list = []

    # 1st block
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # 2nd block
    for i in range(6):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 3rd block
    for i in range(12):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 4th block
    for i in range(24):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 5th block
    for i in range(16):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = keras.layers.GlobalAveragePooling2D()(x)

    output_tensor = keras.layers.Dense(units=num_class, activation="softmax")(keras.layers.Flatten()(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


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

        model = densenet(1000, (224, 224, 3))
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=1)


if __name__ == "__main__":
    go()
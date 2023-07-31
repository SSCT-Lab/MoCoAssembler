import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_.config.paths import DATASETS_PATH


def resnet50(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.ZeroPadding2D((3, 3))(input_tensor)

    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = inceptionA(x, 3, 64, 64, 256, strides=(1, 1))
    x = inceptionB(x, 3, 64, 64, 256)
    x = inceptionB(x, 3, 64, 64, 256)

    x = inceptionA(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)

    x = inceptionA(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)

    x = inceptionA(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)

    x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    output_tensor = keras.layers.Dense(units=class_num, activation="softmax")(keras.layers.Flatten()(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, kernel_size, filters1, filters2, filters3, strides=(2, 2)):
    x = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)
    temp = keras.layers.Conv2D(filters3, (1, 1), strides=strides)(inputs)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    temp = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(temp)

    outputs = keras.layers.add([x, temp])
    return outputs


def inceptionB(inputs, kernel_size, filters1, filters2, filters3):
    x = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    inputs = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(inputs)

    outputs = keras.layers.add([x, inputs])
    return outputs


def go():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=8192)])

    with tf.device("/GPU:0"):
        imagenet = np.load(DATASETS_PATH / "imagenet.npz")
        x_train = imagenet["x_test"][:100]
        y_train = imagenet["y_test"][:100]

    model = resnet50(1000, (224, 224, 3))
    model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
    return model.count_params()


if __name__ == "__main__":
    go()

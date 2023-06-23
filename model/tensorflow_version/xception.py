import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH

def xception(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
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

    output_tensor = keras.layers.Dense(units=class_num, activation="softmax")(keras.layers.Flatten()(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def inceptionA(inputs, filters1, filters2, filters3):
    residual = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False)(inputs)
    residual = keras.layers.BatchNormalization()(residual)

    x = keras.layers.SeparableConv2D(filters=filters2, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=filters3, kernel_size=(3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    residual = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(residual)

    outputs = keras.layers.add([x, residual])
    return outputs


def inceptionB(inputs):
    residual = inputs
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    residual = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(residual)

    outputs = keras.layers.add([x, residual])
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

        model = xception(1000, (224, 224, 3))
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=1)


if __name__ == "__main__":
    go()

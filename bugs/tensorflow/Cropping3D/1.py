import numpy as np
import tensorflow as tf
from moco_tf.config.paths import DATASETS_PATH


def mobilenet(label_num=1000, input_shape=(224, 224, 3)):
    # input layers
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")
    x = input_tensor

    # hidden layers
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation='softplus')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=1, strides=1, padding="valid")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Cropping3D(cropping=-100)(x)

    # output layers
    output_tensor = tf.keras.layers.Flatten()(tf.keras.layers.Dense(units=label_num, activation='softmax')(x))
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
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
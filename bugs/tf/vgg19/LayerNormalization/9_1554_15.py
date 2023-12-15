import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def vgg19(label_num=10, input_shape=(32, 32, 3)):
# vgg19 input layer
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
# vgg19 hidden layer
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", use_bias=True)(input_tensor)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 5))(x)
    x = keras.layers.ZeroPadding2D(padding=1)(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=(1, 2))(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Cropping2D(cropping=(7, 4))(x)
    x = keras.layers.LayerNormalization(epsilon=0.0299392620514376)(x)
    x = keras.layers.LayerNormalization(epsilon=0.0, axis=(4))(x)
# vgg19 output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    cifar10 = np.load(DATASETS_PATH / "cifar10.npz")
    x_train = cifar10['x_train'][:100]
    y_train = cifar10['y_train'][:100]

    x_train= x_train / 255.0

    model = vgg19(10, (32, 32, 3))
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()
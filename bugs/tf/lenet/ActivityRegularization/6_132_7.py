import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def lenet(label_num=10, input_shape=(28, 28, 1)):
# lenet input layer
    input_tensor = keras.Input(shape=input_shape)
# lenet hidden layer
    x = keras.layers.MaxPool2D(strides=1, padding="same", pool_size=(4, 3))(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)
    x = keras.layers.ZeroPadding2D(padding=((3, 1), (5, 6)))(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 4), (6, 3)))(x)
    x = keras.layers.ActivityRegularization(l2=-0.5572025188772234)(x)
# lenet output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def go():
    mnist = np.load(DATASETS_PATH / "mnist.npz")

    x_train = mnist['x_train'][:100]
    y_train = mnist['y_train'][:100]

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

    model = lenet(10, (28, 28, 1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()
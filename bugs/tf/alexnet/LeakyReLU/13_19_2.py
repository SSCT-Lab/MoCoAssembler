import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def alexnet(label_num=10, input_shape=(32, 32, 3)):
# alexnet input layer
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
# alexnet hidden layer
    x = keras.layers.Conv2D(filters=64, kernel_size=(11,11), activation='relu', strides=(5, 1), padding='same')(input_tensor)
    x = keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)
    x = keras.layers.MaxPool2D(padding='same', strides=2)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 3), strides=(2,2))(x)
    x = keras.layers.BatchNormalization(moving_variance_initializer="ones")(x)
    x = keras.layers.Softmax(axis=[2])(x)
    x = keras.layers.AveragePooling2D(padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=(1,1), strides=(1,1), padding="valid")(x)
    x = keras.layers.GlobalAveragePooling2D(keepdims=False)(x)
    x = keras.layers.Dropout(rate=0.5, noise_shape=None)(x)
    x = keras.layers.Layer(trainable=True)(x)
    x = keras.layers.Dropout(rate=0.3915363720848837)(x)
    x = keras.layers.LeakyReLU(alpha=-0.7156048434349785)(x)
# alexnet output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    cifar10 = np.load(DATASETS_PATH / "cifar10.npz")
    x_train = cifar10['x_train'][:100]
    y_train = cifar10['y_train'][:100]

    x_train = x_train / 255.0

    model = alexnet(10, (32, 32, 3))
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()

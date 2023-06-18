import tensorflow as tf
from keras.datasets import mnist
from tensorflow import keras


def lenet(label_num=2, input_shape=(32, 32, 1)):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=200, activation="relu")(x)
    output_tensor = keras.layers.Dense(units=label_num, activation="softmax")(x)

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def go():
    with tf.device("/GPU:0"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

        model = lenet(10, (28, 28, 1))
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=1000, epochs=1, verbose=0)


if __name__ == "__main__":
    go()
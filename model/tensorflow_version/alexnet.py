import tensorflow as tf
from keras.datasets import cifar10
from tensorflow import keras


def alexnet(class_num=10, input_shape=(32, 32, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="same", kernel_initializer="he_normal")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation="relu", padding="same", kernel_initializer="uniform")(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="uniform")(x)

    x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="uniform")(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="uniform")(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation="relu")(x)

    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation="relu")(x)

    # output
    output_tensor = keras.layers.Dense(units=class_num)(x)

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    with tf.device("/GPU:0"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = alexnet(10, (32, 32, 3))
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))


if __name__ == "__main__":
    go()

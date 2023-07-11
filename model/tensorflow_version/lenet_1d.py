from tensorflow import keras


def lenet_1d(label_num=10, input_shape=(28, 28, 1)):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.Conv1D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    x = keras.layers.Flatten()(x)
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))


    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def go():
    model = lenet_1d(10, (28, 1))
    return model.count_params()


if __name__ == "__main__":
    go()
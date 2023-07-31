# lenet function layer

    x = keras.layers.Conv2D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=200, activation="relu")(x)

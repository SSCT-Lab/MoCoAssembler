import tensorflow as tf


def alexnet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=11, activation="relu", strides=4, padding="same")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=5, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)
    x = tf.keras.layers.Dense(units=10, activation="softmax")(x)

    output_tensor = tf.keras.layers.Flatten()(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

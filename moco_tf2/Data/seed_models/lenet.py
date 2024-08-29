import tensorflow as tf


def lenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=200, activation="relu")(x)
    x = tf.keras.layers.Dense(units=10, activation="softmax")(x)

    output_tensor = tf.keras.layers.Flatten()(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

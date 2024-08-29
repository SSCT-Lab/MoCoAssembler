import tensorflow as tf


def mobilenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(input_tensor)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, activation="relu", padding="valid")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1)(x)
    x = tf.keras.layers.Dense(units=1000, activation="softmax")(x)

    output_tensor = tf.keras.layers.Flatten()(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

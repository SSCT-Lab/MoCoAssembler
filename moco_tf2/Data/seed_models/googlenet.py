import tensorflow as tf


def googlenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(inputs=x, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
    x = inception(inputs=x, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(inputs=x, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)

    x = inception(inputs=x, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
    x = inception(inputs=x, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
    x = inception(inputs=x, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)

    x = inception(inputs=x, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = tf.keras.layers.Flatten()(x)

    output_tensor = tf.keras.layers.Dense(units=1000, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
    x1 = tf.keras.layers.Conv2D(filters=ch1x1, kernel_size=1, activation="relu")(inputs)

    x2 = tf.keras.layers.Conv2D(filters=ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x2 = tf.keras.layers.Conv2D(filters=ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")(x2)

    x3 = tf.keras.layers.Conv2D(filters=ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x3 = tf.keras.layers.Conv2D(filters=ch5x5, kernel_size=5, strides=1, padding="same", activation="relu")(x3)

    x4 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
    x4 = tf.keras.layers.Conv2D(filters=pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")(x4)

    outputs = tf.keras.layers.concatenate(inputs=[x1, x2, x3, x4])
    return outputs

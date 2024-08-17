import tensorflow as tf


def googlenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(x, 64, 96, 128, 16, 32, 32)
    x = inception(x, 128, 128, 192, 32, 96, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = inception(x, 192, 96, 208, 16, 48, 64)

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    output_tensor = tf.keras.layers.Dense(units=1000, activation="softmax")(tf.keras.layers.Flatten()(x))

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

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x1)
    x2 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x2)
    x3 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x3)
    x4 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x4)

    outputs = tf.keras.layers.concatenate([x1, x2, x3, x4])
    return outputs
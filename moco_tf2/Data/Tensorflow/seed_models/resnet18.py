import tensorflow as tf


def resnet18(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="valid", activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = inceptionB(x, filters=64, kernel_size=3, strides=1, padding="same")
    x = inceptionB(x, filters=64, kernel_size=3, strides=1, padding="same")

    x = inceptionA(x, filters=128, kernel_size=3, strides=2, padding="same")
    x = inceptionB(x, filters=128, kernel_size=3, strides=1, padding="same")

    x = inceptionA(x, filters=256, kernel_size=3, strides=2, padding="same")
    x = inceptionB(x, filters=256, kernel_size=3, strides=1, padding="same")

    x = inceptionA(x, filters=512, kernel_size=3, strides=2, padding="same")
    x = inceptionB(x, filters=512, kernel_size=3, strides=1, padding="same")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output_tensor = tf.keras.layers.Dense(units=1000, activation="softmax")(tf.keras.layers.Flatten()(x))

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, filters=64, kernel_size=3, strides=1, padding="same"):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(x)

    temp = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding="same", activation="relu")(inputs)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    temp = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(temp)
    outputs = tf.keras.layers.add([x, temp])
    return outputs


def inceptionB(inputs, filters=64, kernel_size=3, strides=1, padding="same"):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(x)

    # reshape
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x)
    inputs = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(inputs)

    outputs = tf.keras.layers.add([x, inputs])
    return outputs
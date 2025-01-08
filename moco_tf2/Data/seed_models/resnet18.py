import tensorflow as tf


def resnet18(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="valid", activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = inceptionB(inputs=x, filters=64, kernel_size=3, strides=1, padding="same")
    x = inceptionB(inputs=x, filters=64, kernel_size=3, strides=1, padding="same")

    x = inceptionA(inputs=x, filters=128, kernel_size=3, strides=2, padding="same")
    x = inceptionB(inputs=x, filters=128, kernel_size=3, strides=1, padding="same")

    x = inceptionA(inputs=x, filters=256, kernel_size=3, strides=2, padding="same")
    x = inceptionB(inputs=x, filters=256, kernel_size=3, strides=1, padding="same")

    x = inceptionA(inputs=x, filters=512, kernel_size=3, strides=2, padding="same")
    x = inceptionB(inputs=x, filters=512, kernel_size=3, strides=1, padding="same")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    output_tensor = tf.keras.layers.Dense(units=1000, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, filters, kernel_size, strides, padding):
    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(x1)

    x2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding="same", activation="relu")(inputs)

    outputs = tf.keras.layers.add(inputs=[x1, x2])
    return outputs


def inceptionB(inputs, filters, kernel_size, strides, padding):
    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inputs)
    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(x1)

    outputs = tf.keras.layers.add(inputs=[x1, inputs])
    return outputs

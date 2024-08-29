import tensorflow as tf


def squeezenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=96,  activation="relu", kernel_size=7, strides=2, padding="same")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = inception(inputs=x, s1=16, e1=64, e3=64)
    x = inception(inputs=x, s1=16, e1=64, e3=64)
    x = inception(inputs=x, s1=32, e1=128, e3=128)
    x = inception(inputs=x, s1=32, e1=128, e3=128)
    x = inception(inputs=x, s1=48, e1=192, e3=192)
    x = inception(inputs=x, s1=48, e1=192, e3=192)
    x = inception(inputs=x, s1=64, e1=256, e3=256)
    x = inception(inputs=x, s1=64, e1=256, e3=256)

    x = tf.keras.layers.Conv2D(filters=5, activation="relu", kernel_size=1, strides=1, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=1000, activation="softmax")(x)

    output_tensor = tf.keras.layers.Flatten()(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs, s1, e1, e3):
    x = tf.keras.layers.Conv2D(filters=s1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(inputs)
    y1 = tf.keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = tf.keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    outputs = tf.keras.layers.concatenate(inputs=[y1, y2])
    return outputs

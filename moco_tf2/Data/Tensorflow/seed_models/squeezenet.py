import tensorflow as tf


def squeezenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=96,  activation="relu", kernel_size=(7, 7), strides=2, padding="same")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = inception(x, s1=16, e1=64, e3=64)
    x = inception(x, s1=16, e1=64, e3=64)
    x = inception(x, s1=32, e1=128, e3=128)
    x = inception(x, s1=32, e1=128, e3=128)
    x = inception(x, s1=48, e1=192, e3=192)
    x = inception(x, s1=48, e1=192, e3=192)
    x = inception(x, s1=64, e1=256, e3=256)
    x = inception(x, s1=64, e1=256, e3=256)

    x = tf.keras.layers.Conv2D(filters=5, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output_tensor = tf.keras.layers.Flatten()(tf.keras.layers.Dense(units=1000, activation="softmax")(x))
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(x, s1, e1, e3):
    x = tf.keras.layers.Conv2D(filters=s1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = tf.keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = tf.keras.layers.concatenate([y1, y2])
    return outputs
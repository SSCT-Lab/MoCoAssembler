import tensorflow as tf


def alexnet_2(im_height=224, im_width=224, num_classes=1000):
    input_image = tf.keras.layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = tf.keras.layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)
    x = tf.keras.layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    predict = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=predict)
    return model




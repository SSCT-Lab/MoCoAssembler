import tensorflow as tf
from tensorflow import keras


def vgg16(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    # 1st block
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same")(input_tensor)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 2nd block
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 3rd block
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 4th block
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # 5th block
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    # full connection
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=4096, activation="relu")(x)

    x = keras.layers.Dense(units=4096, activation="relu")(x)
    x = keras.layers.Dense(units=class_num, activation="softmax")(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    net = vgg16(class_num=1000, input_shape=(224, 224, 3))

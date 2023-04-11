import tensorflow as tf
from tensorflow.python.keras import Input, layers, models


def VGG16(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = Input(shape=input_shape, dtype="float32")

    # 1st block
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1a")(input_tensor)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1b")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)

    # 2nd block
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same", name="conv2a")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same", name="conv2b")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)

    # 3rd block
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3a")(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3b")(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same", name="conv3c")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)

    # 4th block
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4a")(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4b")(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv4c")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool4")(x)

    # 5th block
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5a")(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5b")(x)
    x = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same", name="conv5c")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool5")(x)

    # full connection
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu",  name="fc6")(x)

    x = layers.Dense(4096, activation="relu", name="fc7")(x)
    x = layers.Dense(class_num, activation="softmax", name="fc8")(x)

    output_tensor = x

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    net = VGG16(class_num=1000, input_shape=(224, 224, 3))
    net.summary()

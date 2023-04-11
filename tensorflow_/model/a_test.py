import tensorflow as tf
from tensorflow import keras

def NET(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1a")(input_tensor)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same", name="conv1b")(x)
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",  name="fc6")(x)

    x = keras.layers.Dense(100, activation="relu", name="fc7")(x)

    x = keras.layers.Dense(10, activation="softmax", name="fc8")(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    net = NET(class_num=1000, input_shape=(224, 224, 3))
    net.summary()

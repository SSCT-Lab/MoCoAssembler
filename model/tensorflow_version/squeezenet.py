import tensorflow as tf
from tensorflow import keras


def squeezenet(input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.Conv2D(filters=96,  activation="relu", kernel_size=(7, 7), strides=2, padding="same")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # 2nd block
    x = inception(x, s1=16, e1=64, e3=64)

    # 3nd block
    x = inception(x, s1=16, e1=64, e3=64)

    # 4nd block
    x = inception(x, s1=32, e1=128, e3=128)

    # 5nd block
    x = inception(x, s1=32, e1=128, e3=128)

    # 6nd block
    x = inception(x, s1=48, e1=192, e3=192)

    # 7nd block
    x = inception(x, s1=48, e1=192, e3=192)

    # 8nd block
    x = inception(x, s1=64, e1=256, e3=256)

    # 9nd block
    x = inception(x, s1=64, e1=256, e3=256)

    # 10th block
    x = keras.layers.Conv2D(filters=5, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.nn.softmax(x)

    # output
    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(x, s1, e1, e3):
    x = keras.layers.Conv2D(filters=s1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    shape = tf.shape(x)
    y1 = tf.reshape(y1, shape)
    y2 = tf.reshape(y2, shape)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


if __name__ == '__main__':
    model = squeezenet((224, 224, 3))
    model.summary()

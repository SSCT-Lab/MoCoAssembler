import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def squeezenet(label_num=1000, input_shape=(224, 224, 3)):
# squeezenet input layer
    input_tensor = keras.Input(shape=input_shape)
# squeezenet hidden layer
    x = keras.layers.Conv2D(filters=631, activation="relu", kernel_size=(7,7), strides=2, padding="same")(input_tensor)
    x = keras.layers.AveragePooling2D(pool_size=(3,3), strides=2, padding="valid")(x)
    x = inception_1(x, s1=16, e1=64, e3=64)
    x = inception_2(x, s1=16, e1=64, e3=64)
    x = inception_3(x, s1=32, e1=128, e3=128)
    x = inception_4(x, s1=32, e1=128, e3=128)
    x = inception_5(x, s1=48, e1=192, e3=192)
# squeezenet output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    imagenet = np.load(DATASETS_PATH / "imagenet.npz")
    x_train = imagenet['x_test'][:100]
    y_train = imagenet['y_test'][:100]

    model = squeezenet(1000, (224, 224, 3))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
    return model.count_params()


def inception_1(x, s1, e1, e3):
    x = keras.layers.SeparableConv2D(filters=s1, activation="relu", kernel_size=(1,1), strides=1, padding="same", depthwise_constraint=None)(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


def inception_2(x, s1, e1, e3):
    x = keras.layers.Conv2D(filters=s1, activation="elu", kernel_size=(1,1), strides=1, padding="same")(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


def inception_3(x, s1, e1, e3):
    x = keras.layers.Conv2D(filters=s1, activation="relu", kernel_size=(1,1), strides=1, padding="same")(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


def inception_4(x, s1, e1, e3):
    x = keras.layers.Conv2D(filters=s1, activation="relu", kernel_size=(1,1), strides=3, padding="same")(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="relu", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


def inception_5(x, s1, e1, e3):
    x = keras.layers.DepthwiseConv2D(activation="relu", kernel_size=(1,1), strides=1, padding="same", use_bias=True)(x)
    y1 = keras.layers.Conv2D(filters=e1, activation="relu", kernel_size=(1, 1), strides=1, padding="same")(x)
    y2 = keras.layers.Conv2D(filters=e3, activation="serialize", kernel_size=(3, 3), strides=1, padding="same")(x)

    # reshape
    target_height = x.shape[1]
    target_width = x.shape[2]
    y1 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y1)
    y2 = keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(y2)

    outputs = keras.layers.concatenate([y1, y2])
    return outputs


if __name__ == "__main__":
    go()

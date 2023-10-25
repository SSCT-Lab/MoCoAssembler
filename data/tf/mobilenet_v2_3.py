import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6.0)


def _conv2d(inputs, filters, kernel_size, strides):
    channel_axis = 1 if K.image_data_format == "channels_first" else -1

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    return tf.keras.layers.Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel_size, strides, t, r, alpha=1):
    channel_axis = 1 if K.image_data_format == "channels_first" else -1

    up_channel = K.int_shape(inputs)[1] * t if K.image_data_format == "channels_first" else K.int_shape(inputs)[-1] * t

    down_channel = filters * alpha

    x = _conv2d(inputs=inputs, filters=up_channel, kernel_size=(1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=1)(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(relu6)(x)

    x = tf.keras.layers.Conv2D(filters=down_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    if r:
        x = tf.keras.layers.Add()([inputs, x])

    return x


def _inverted_residual_block(inputs, filters, kernel_size, strides, t, n, alpha=1, r=False):
    x = _bottleneck(inputs, filters, kernel_size, strides, t, False, alpha)
    for _ in range(1, n):
        x = _bottleneck(x, filters, kernel_size, 1, t, True, alpha)

    return x


def MobilenetV2(input_shape, k, alpha=1):
    inputs = Input(shape=input_shape)

    x = _conv2d(inputs, filters=32 * alpha, kernel_size=3, strides=2)
    x = _inverted_residual_block(inputs=x, filters=16, kernel_size=3, strides=1, t=1, n=1)
    x = _inverted_residual_block(inputs=x, filters=24, kernel_size=3, strides=2, t=6, n=2)
    x = _inverted_residual_block(inputs=x, filters=32, kernel_size=3, strides=2, t=6, n=3)
    x = _inverted_residual_block(inputs=x, filters=64, kernel_size=3, strides=2, t=6, n=4)
    x = _inverted_residual_block(inputs=x, filters=96, kernel_size=3, strides=1, t=6, n=3)
    x = _inverted_residual_block(inputs=x, filters=160, kernel_size=3, strides=2, t=6, n=3)
    x = _inverted_residual_block(inputs=x, filters=320, kernel_size=3, strides=1, t=6, n=1)
    x = _conv2d(inputs=x, filters=1280 * alpha, kernel_size=(1, 1), strides=1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = tf.keras.layers.Conv2D(filters=k, kernel_size=1, strides=1, padding='same', activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def mobilenet_v2_3():
    model = MobilenetV2(input_shape=(224, 224, 3), k=1000)
    return model

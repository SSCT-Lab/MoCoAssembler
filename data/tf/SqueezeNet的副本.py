import tensorflow as tf
from tensorflow.keras import Sequential, Model


class Fire(tf.keras.layers.Layer):
    def __init__(self, out_channels, squeeze_channel):
        super(Fire, self).__init__()
        self.squeeze = Sequential([
            tf.keras.layers.Conv2D(squeeze_channel, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.expand_1x1 = Sequential([
            tf.keras.layers.Conv2D(int(out_channels / 2), (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.expand_3x3 = Sequential([
            tf.keras.layers.Conv2D(int(out_channels / 2), (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x, training=False):
        x = self.squeeze(x, training=training)
        x = tf.concat([
            self.expand_1x1(x, training=training),
            self.expand_3x3(x, training=training)
        ], -1)

        return x


class SqueezeNet(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(SqueezeNet, self).__init__()
        self.stem = Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(96, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        ])
        self.fire = Sequential([
            Fire(128, 16),
            Fire(128, 16),
            Fire(256, 32),
            Fire(256, 32),
            Fire(384, 48),
            Fire(384, 48),
            Fire(512, 64),
            Fire(512, 64)
        ])
        self.conv = tf.keras.layers.Conv2D(num_classes, 1)
        self.ap = tf.keras.layers.AveragePooling2D((7, 7), strides=1)
        self.mp = tf.keras.layers.MaxPooling2D()
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.fire(x, training=training)
        x = self.conv(x, training=training)
        x = self.ap(x)
        x = self.mp(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def squeezenet(num_classes):
    return SqueezeNet(num_classes)

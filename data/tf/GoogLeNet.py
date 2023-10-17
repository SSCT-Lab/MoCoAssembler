import tensorflow as tf
from tensorflow.keras import Sequential, Model


class Inception(tf.keras.layers.Layer):
    def __init__(self, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()

        self.b1 = Sequential([
            tf.keras.layers.Conv2D(n1x1, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.b2 = Sequential([
            tf.keras.layers.Conv2D(n3x3_reduce, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n3x3, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.b3 = Sequential([
            tf.keras.layers.Conv2D(n5x5_reduce, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n5x5, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n5x5, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.b4 = Sequential([
            tf.keras.layers.MaxPool2D((3, 3), 1, padding='same'),
            tf.keras.layers.Conv2D(pool_proj, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

    def call(self, x):
        x = tf.concat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], axis=3)
        return x


class GoogleNet(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(GoogleNet, self).__init__()
        self.layer1 = Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(192, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer2 = Sequential([
            Inception(64, 96, 128, 16, 32, 32),
            Inception(128, 128, 192, 32, 96, 64),
            tf.keras.layers.MaxPool2D((3, 3), 2, padding='same'),
        ])
        self.layer3 = Sequential([
            Inception(192, 96, 208, 16, 48, 64),
            Inception(160, 112, 224, 24, 64, 64),
            Inception(128, 128, 256, 24, 64, 64),
            Inception(112, 144, 288, 32, 64, 64),
            Inception(256, 160, 320, 32, 128, 128),
            tf.keras.layers.MaxPool2D((3, 3), 2, padding='same'),
        ])
        self.layer4 = Sequential([
            Inception(256, 160, 320, 32, 128, 128),
            Inception(384, 192, 384, 48, 128, 128)
        ])
        self.layer5 = Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4),
        ])
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.layer1(inputs, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        return x


def GoogLeNet(num_classes):
    return GoogleNet(num_classes)

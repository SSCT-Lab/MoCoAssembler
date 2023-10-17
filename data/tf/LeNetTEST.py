import tensorflow as tf
from tensorflow.keras import Model


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid')
        self.a1 = tf.keras.layers.Activation('sigmoid')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid')
        self.a2 = tf.keras.layers.Activation('sigmoid')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.flatten3 = tf.keras.layers.Flatten()

        self.d4 = tf.keras.layers.Dense(120, activation='sigmoid')

        self.d5 = tf.keras.layers.Dense(84, activation='sigmoid')

        self.d6 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.flatten3(x)
        x = self.d4(x)
        x = self.d5(x)
        y = self.d6(x)
        return y


model = LeNet5()

import tensorflow as tf

from typing import Sequence, Union, Callable, Optional


class DropConnectDense(tf.keras.layers.Layer):
    def __init__(self, units: int, drop_rate: float):
        super(DropConnectDense, self).__init__()
        self.units = units
        self.drop_rate = drop_rate

        self.dropout = tf.keras.layers.Dropout(self.drop_rate)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel', shape=[int(input_shape[-1]), self.units], trainable=True
        )
        self.bias = self.add_weight(
            'bias', shape=[self.units, ], trainable=True
        )

    def call(self, inputs, training: bool):
        y = tf.matmul(inputs, self.dropout(self.kernel, training=training))
        y += self.dropout(self.bias, training=training)
        return tf.nn.relu(y)


# NOTE !!!
# There is 2 major problem to implement this model.
# 1. The paper says that zero padding is not done in the second convolution, but this does not match the spatial size.
#    So I just use zero padding for all Conv.
# 2. There is no specific description that where activation function is placed. After Conv? BN? Dropout?
#    So I just use it after Conv/FC.
class EnsBlock(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: Sequence[int], drop_rate: float,
            down_sample: bool = True, drop_after: bool = False
    ):
        super(EnsBlock, self).__init__()

        self.forward = tf.keras.Sequential([
                tf.keras.layers.Conv2D(n_filters[0], 3, padding='same', activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(drop_rate),
                tf.keras.layers.Conv2D(n_filters[1], 3, padding='same', activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(drop_rate),
                tf.keras.layers.Conv2D(n_filters[2], 3, padding='same', activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization()
        ])
        if down_sample:
            self.forward.add(tf.keras.layers.MaxPool2D(2, 2))
        if drop_after:
            self.forward.add(tf.keras.layers.Dropout(drop_rate))

    def call(self, inputs, training: bool = False):
        return self.forward(inputs, training=training)


# NOTE !!!
# There is no specific description that how to extract features from the last layer. Flatten? GAP?
# So I just use GAP.
class Classifier(tf.keras.layers.Layer):
    def __init__(self, n_classes: int, drop_rate: float):
        super(Classifier, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(drop_rate),
            DropConnectDense(512, drop_rate),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

    def call(self, inputs, training: bool = False):
        return self.forward(inputs, training=training)

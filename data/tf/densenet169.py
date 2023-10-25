import tensorflow as tf


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(inner_channel, (1, 1), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)
        ])

    def call(self, x, training=False):
        return tf.concat([x, self.bottle_neck(x, training=training)], axis=-1)


class Transition(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(Transition, self).__init__()

        self.down_sample = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(out_channels, (1, 1), use_bias=False),
            tf.keras.layers.AveragePooling2D((2, 2), strides=2)
        ])

    def call(self, x, training=False):
        return self.down_sample(x, training=training)


class DenseNet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block,
                 nblocks,
                 growth_rate=12,
                 reduction=0.5,
                 input_shape=(32, 32, 3)):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(inner_channels, (3, 3),
                          padding='same', use_bias=False)
        ])

        self.features = tf.keras.Sequential()

        for idx in range(len(nblocks) - 1):
            self.features.add(block(nblocks[idx]))
            inner_channels += growth_rate * nblocks[idx]

            out_channels = int(reduction * inner_channels)
            self.features.add(Transition(out_channels))
            inner_channels = out_channels

        self.features.add(self._make_dense_layers(
            block, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add(tf.keras.layers.BatchNormalization())
        self.features.add(tf.keras.layers.ReLU())

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def _make_dense_layers(self, block, nblocks):
        dense_block = tf.keras.Sequential()
        for idx in range(nblocks):
            dense_block.add(block(self.growth_rate))
        return dense_block

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.features(x, training=training)
        x = self.gap(x)
        x = self.fc(x)
        return x


def densenet169(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 32, 32], growth_rate=32)

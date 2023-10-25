import tensorflow as tf
from keras import Sequential, Model


class BasicResidualSEBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides, r=16):
        super(BasicResidualSEBlock, self).__init__()

        self.residual = Sequential([
            tf.keras.layers.Conv2D(out_channels, (3, 3),
                                   strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_channels * self.expansion,
                                   (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.shortcut = Sequential()
        if strides != 1 or in_channels != out_channels * self.expansion:
            self.shortcut.add(tf.keras.layers.Conv2D(out_channels * self.expansion,
                                                     (1, 1),
                                                     strides=strides))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation = Sequential([
            tf.keras.layers.Dense(out_channels * self.expansion //
                                  4, activation='relu'),
            tf.keras.layers.Dense(out_channels * self.expansion, activation='sigmoid')
        ])

    def call(self, x, training=False, **kwargs):
        shortcut = self.shortcut(x, training=training)
        residual = self.residual(x, training=training)

        squeeze = self.squeeze(residual, training=training)
        excitation = self.excitation(squeeze, training=training)
        excitation = tf.reshape(excitation, (-1, 1, 1, residual.shape[-1]))
        x = residual * tf.broadcast_to(excitation, residual.shape) + shortcut

        return tf.nn.relu(x)


class BottleneckResidualSEBlock(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides, r=16):
        super(BottleneckResidualSEBlock, self).__init__()

        self.residual = Sequential([
            tf.keras.layers.Conv2D(out_channels, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_channels, (3, 3),
                                   strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_channels * self.expansion, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation = Sequential([
            tf.keras.layers.Dense(out_channels * self.expansion //
                                  r, activation='relu'),
            tf.keras.layers.Dense(out_channels * self.expansion, activation='sigmoid')
        ])

        self.shortcut = Sequential()
        if strides != 1 or in_channels != out_channels * self.expansion:
            self.shortcut.add(tf.keras.layers.Conv2D(out_channels * self.expansion,
                                                     (1, 1),
                                                     strides=strides))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

    def call(self, x, training=False, **kwargs):
        shortcut = self.shortcut(x, training=training)

        residual = self.residual(x, training=training)
        squeeze = self.squeeze(residual, training=training)
        excitation = self.excitation(squeeze)
        excitation = tf.reshape(excitation, (-1, 1, 1, residual.shape[-1]))
        x = residual * tf.broadcast_to(excitation, residual.shape) + shortcut

        return tf.nn.relu(x)


class SEResNet(Model):
    def __init__(self, block, block_num, num_classes, input_shape=(32, 32, 3)):
        super(SEResNet, self).__init__()

        self.in_channels = 64

        self.front = Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 516, 2)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def _make_stage(self, block, num, out_channels, strides):
        nets = []
        nets.append(block(self.in_channels, out_channels, strides))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            nets.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return Sequential(nets)

    def call(self, inputs, training=False, **kwargs):
        x = self.front(inputs, training=training)

        x = self.stage1(x, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)

        x = self.gap(x)
        x = self.fc(x)

        return x


def se_resnet_50_2():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3], 10)

import tensorflow as tf
from tensorflow.keras import Sequential


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        identity = self.downsample(inputs)

        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.stem = Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('relu'),
                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()

        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])

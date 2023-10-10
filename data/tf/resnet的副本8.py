import tensorflow as tf

class BlockSet(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(BlockSet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='SAME')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='SAME')
        self.bn2 = tf.keras.layers.BatchNormalization()
        if strides != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters, (1, 1), strides=strides))  # 这里不选择池化可能是希望下采样后更接近原来
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        out = tf.keras.layers.add([identity, out])
        out = tf.nn.relu(out)

        return out


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, out_class):
        super(ResNet, self).__init__()
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'),
                                         tf.keras.layers.BatchNormalization(),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1, 1), padding='same')])
        self.block1 = self.BuildBlock(64, layer_dims[0], 1)
        self.block2 = self.BuildBlock(128, layer_dims[1], 2)
        self.block3 = self.BuildBlock(256, layer_dims[2], 2)
        self.block4 = self.BuildBlock(512, layer_dims[3], 2)
        self.full = tf.keras.layers.Conv2D(512, (4, 4), strides=1, padding='valid')
        self.classier = tf.keras.layers.Conv2D(out_class, (1, 1), strides=1, padding='valid')

    def BuildBlock(self, filters, blocks, strides=1):
        res_block = tf.keras.Sequential()
        res_block.add(BlockSet(filters, strides))
        for i in range(1, blocks):
            res_block.add(BlockSet(filters, 1))
        return res_block

    def call(self, inputs, training=False):
        out = self.stem(inputs)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.full(out)
        out = self.classier(out)
        out = tf.keras.layers.Flatten()(out)
        return out




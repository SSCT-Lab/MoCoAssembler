import tensorflow as tf
from keras import layers, Model, Sequential


def conv3x3(kernels, strides=1):
    return layers.Conv2D(kernels, 3, strides=strides, padding='same', use_bias=False)


def conv1x1(kernels, strides=1):
    return layers.Conv2D(kernels, 1, strides=strides, use_bias=False)


class BasicBlock(Model):

    expansion = 1

    def __init__(self,
                 kernels,
                 strides=1,
                 downsample=None,
                 norm_layer=None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer()

        self.conv1 = conv3x3(kernels, strides)
        self.bn1 = norm_layer()
        self.relu = layers.ReLU()
        self.conv2 = conv3x3(kernels)

        if start_block or end_block:
            self.bn2 = norm_layer()

        self.downsample = downsample
        self.strides = strides

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x, training=False, **kwargs):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x, training=training)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn2(out, training=training)
            out = self.relu(out)

        return out


class Bottleneck(Model):
    expansion = 4

    def __init__(self,
                 kernels,
                 strides=1,
                 downsample=None,
                 norm_layer=None,
                 start_block=False,
                 end_block=False,
                 exclude_bn0=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer()

        self.conv1 = conv1x1(kernels)
        self.bn1 = norm_layer()
        self.conv2 = conv3x3(kernels, strides)
        self.bn2 = norm_layer()
        self.conv3 = conv1x1(kernels * self.expansion)

        if start_block or end_block:
            self.bn3 = norm_layer()

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.strides = strides

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x, training=False, **kwargs):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class iResNet(Model):
    def __init__(self,
                 block,
                 layer,
                 num_classes,
                 input_shape,
                 zero_init_residual=False,
                 norm_layer=None,
                 dropout_prob0=0.0):
        super(iResNet, self).__init__()
        self.inplanes = 64
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        self.conv1 = layers.Conv2D(
            64, 7, strides=2, padding='same', use_bias=False, input_shape=input_shape)
        self.bn1 = norm_layer()
        self.relu = layers.ReLU()
        self.layer1 = self._make_layer(
            block, 64, layer[0], strides=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, 128, layer[1], strides=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(
            block, 256, layer[2], strides=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(
            block, 512, layer[3], strides=2, norm_layer=norm_layer)
        self.gap = layers.GlobalAveragePooling2D()

        if dropout_prob0 > 0.0:
            self.dp = layers.Dropout(dropout_prob0)
        else:
            self.dp = None

        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_layer(self, block, kernels, blocks, strides=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        downsample = None
        if strides != 1 and self.inplanes != kernels * block.expansion:
            downsample = Sequential([
                layers.ZeroPadding2D((1, 1)),
                layers.MaxPooling2D(3, strides=strides),
                conv1x1(kernels * block.expansion),
                norm_layer()
            ])
        elif self.inplanes != kernels * block.expansion:
            downsample = Sequential([
                conv1x1(kernels * block.expansion),
                norm_layer
            ])
        elif strides != 1:
            downsample = Sequential([
                layers.ZeroPadding2D((1, 1)),
                layers.MaxPooling2D(3, strides=strides)
            ])

        nets = []
        nets.append(block(kernels, strides, downsample,
                          norm_layer, start_block=True))
        self.inplanes = kernels * block.expansion
        exclude_bn0 = True
        for _ in range(1, (blocks-1)):
            nets.append(block(kernels, norm_layer=norm_layer,
                              exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        nets.append(block(kernels, norm_layer=norm_layer,
                          end_block=True, exclude_bn0=exclude_bn0))

        return Sequential(nets)

    def call(self, x, training=False, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.gap(x)

        if self.dp is not None:
            x = self.dp(x, training=training)

        x = self.fc(x)

        return x


def iresnet152_1(input_shape=(224, 224, 3)):
    iresnet = iResNet(Bottleneck, [3, 8, 36, 3], 10, input_shape)
    inputs_ = tf.keras.Input(shape=input_shape)
    res = iresnet(inputs_, training=True)
    model = tf.keras.Model(inputs_, res)
    return model

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from blocks import BasicBlock, Bottleneck, conv1x1, conv3x3


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

    def call(self, x, training=False):
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


def iresnet18(num_classes, input_shape):
    return iResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_shape)


def iresnet34(num_classes, input_shape):
    return iResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_shape)


def iresnet50(num_classes, input_shape):
    return iResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_shape)


def iresnet101(num_classes, input_shape):
    return iResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_shape)


def iresnet152(num_classes, input_shape):
    return iResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_shape)


def iresnet200(num_classes, input_shape):
    return iResNet(Bottleneck, [3, 24, 36, 3], num_classes, input_shape)


def iresnet302(num_classes, input_shape):
    return iResNet(Bottleneck, [4, 34, 58, 4], num_classes, input_shape)


def iresnet404(num_classes, input_shape):
    return iResNet(Bottleneck, [4, 46, 80, 4], num_classes, input_shape)


def iresnet1001(num_classes, input_shape):
    return iResNet(Bottleneck, [4, 155, 170, 4], num_classes, input_shape)

import tensorflow as tf
from tensorflow.keras import layers

from config import _BuildConfig

def _GetStem(stem_type):
    stems = {
        'resnet_stem_cifar': ResNetStemCifar,
        'resnet_stem_imagenet': ResNetStemImagenet,
        'simple_stem_imagenet': SimpleStemImagenet
    }

    assert stem_type in stems.keys(), 'Stem type {} is not supported.'.format(stem_type)

    return stems[stem_type]

def _GetBlock(block_type):
    blocks = {
        'vanilla_block': VanillaBlock,
        'residual_basic_block': ResidualBasicBlock,
        'residual_bottleneck_block': ResidualBottleneckBlock
    }

    assert block_type in blocks.keys(), 'Block type {} is not supported.'.format(block_type)

    return blocks[block_type]

def Conv2D(widths, kernel_size, stride, padding, use_bias=False):

    return layers.Conv2D(
        widths,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, 'fan_out'),
    )

BN_PARAMS = {'momentum': None, 'epsilon': None}
def BatchNormalization():
    return layers.BatchNormalization(momentum=BN_PARAMS['momentum'], epsilon=BN_PARAMS['epsilon'])

class ResNetStemCifar(layers.Layer):
    """ ResNet stem for CIFAR """

    def __init__(self, stem_widths, name='stem'):
        super(ResNetStemCifar, self).__init__(name=name)

        # 3x3 Conv, BN, ReLU
        self.stem = tf.keras.Sequential([
            Conv2D(stem_widths, 3, 1, 'same'),
            BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=True):
        return self.stem(x, training=training)

class ResNetStemImagenet(layers.Layer):
    """ ResNet stem for Imagenet """

    def __init__(self, stem_widths, name='stem'):
        super(ResNetStemImagenet, self).__init__(name=name)

        # 7x7 Conv, BN, ReLU, maxpool
        self.stem = tf.keras.Sequential([
            Conv2D(stem_widths, 7, 2, 'same'),
            BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        ])

    def call(self, x, training=True):
        return self.stem(x, training=training)

class SimpleStemImagenet(layers.Layer):
    """ Simple stem for Imagenet """

    def __init__(self, stem_widths, name='stem'):
        super(SimpleStemImagenet, self).__init__(name=name)

        # 3x3 Conv, BN, ReLU
        self.stem = tf.keras.Sequential([
            Conv2D(stem_widths, 3, 2, 'same'),
            BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=True):
        return self.stem(x, training=training)

class VanillaBlock(layers.Layer):
    """ Vanilla Block: [3x3 Conv, BN, ReLU] x 2 """

    def __init__(self, in_widths, out_widths, stride, bottleneck_ratio=None, num_groups=None, se_ratio=None, name='vanilla_block'):
        super(VanillaBlock, self).__init__(name=name)

        assert (
            bottleneck_ratio is None and num_groups is None and se_ratio is None
        ), 'vanilla block does not support bottleneck layers, group convolutions or squueze and excitation.'

        self.block = tf.keras.Sequential([
            # 3x3 Conv, BN, ReLU
            Conv2D(widths, 3, stride, 'same'),
            layers.BatchNormalization(momentum=config.BN.MOMENTUM, epsilon=config.BN.EPS),
            BatchNormalization(),
            layers.ReLU(),
            # 3x3 Conv, BN, ReLU
            Conv2D(widths, 3, 1, 'same'),
            BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=True):
        return self.block(x, training=training)

class ResidualBasicBlock(layers.Layer):
    """ Residual basic block x + F(x), F = [3x3 Conv, BN, ReLU] x 2 """

    def __init__(self, in_widths, out_widths, stride, projection, bottleneck_ratio=None, num_groups=None, se_ratio=None, name='residual_basic_block'):
        super(ResidualBasicBlock, self).__init__(name=name)

        assert (
            bottleneck_ratio is None and num_groups is None and se_ratio is None
        ), 'Residual basic block does not support bottleneck layers, group convolutions or squueze and excitation.'

        self.block = tf.keras.Sequential([
            # 3x3 Conv, BN, ReLU
            Conv2D(out_widths, 3, stride, 'same'),
            BatchNormalization(),
            layers.ReLU(),
            # 3x3 Conv, BN
            Conv2D(out_widths, 3, 1, 'same'),
            BatchNormalization()
        ])

        if projection:
            self.residual_connection = tf.keras.Sequential([
                Conv2D(out_widths, 1, stride, 'valid'),
                BatchNormalization()
            ]) 
        else:
            self.residual_connection = layers.Activation('linear')

        self.relu = layers.ReLU()

    def call(self, x, training=True):
        x = self.block(x, training=training) + self.residual_connection(x, training=training)
        x = self.relu(x)

        return x

class GroupConv2D(layers.Layer):
    def __init__(self, widths, stride, num_groups, name='group_conv_2d'):
        super(GroupConv2D, self).__init__(name=name)

        # g: number of groups
        self.num_groups = num_groups
        assert widths % self.num_groups == 0, 'Number of widths are not divisble by g.'
        group_widths = widths // self.num_groups

        self.convs = [Conv2D(group_widths, 3, stride, 'same') for _ in range(self.num_groups)]

    def call(self, x, training=True):
        x_groups = tf.split(x, num_or_size_splits=self.num_groups, axis=-1)
        x_groups = [conv(x_, training=training) for x_, conv in zip(x_groups, self.convs)]
        x = tf.concat(x_groups, axis=-1)

        return x

class SqueezeExcitation(layers.Layer):
    def __init__(self, widths, se_widths, name='squeeze_excitation'):
        super(SqueezeExcitation, self).__init__(name=name)

        self.se = tf.keras.Sequential([
            Conv2D(se_widths, 1, 1, 'valid', True),
            layers.ReLU(),
            Conv2D(widths, 1, 1, 'valid', True),
            layers.Activation('sigmoid')
        ])

    def call(self, x, training=True):
        return x * self.se(tf.reduce_mean(x, [1, 2], keepdims=True))

class ResidualBottleneckBlock(layers.Layer):
    """ Residual bottleneck block: x + F(x), F = [1x1 Conv, 3x3 Conv, 1x1 Conv] """

    def __init__(self, in_widths, out_widths, stride, projection, bottleneck_ratio=1.0, num_groups=1, se_ratio=0.25, name='residual_bottleneck_block'):
        super(ResidualBottleneckBlock, self).__init__(name=name)

        bottleneck_widths = int(round(out_widths*bottleneck_ratio))
        if se_ratio:
            se_widths = int(round(in_widths*se_ratio))

        self.block = tf.keras.Sequential([
            # 1x1 Conv, BN, ReLU
            Conv2D(bottleneck_widths, 1, 1, 'valid'),
            BatchNormalization(),
            layers.ReLU(),
            # 3x3 Conv, BN, ReLU
            GroupConv2D(bottleneck_widths, stride, num_groups),
            BatchNormalization(),
            layers.ReLU(),
            # SE
            SqueezeExcitation(bottleneck_widths, se_widths) if se_ratio else layers.Activation('linear'),
            # 1x1 Conv, BN, ReLU
            Conv2D(out_widths, 1, 1, 'valid'),
            BatchNormalization()
        ])

        if projection:
            self.residual_connection = tf.keras.Sequential([
                Conv2D(out_widths, 1, stride, 'valid'),
                BatchNormalization()
            ]) 
        else:
            self.residual_connection = layers.Activation('linear')

        self.relu = layers.ReLU()

    def call(self, x, training=True):
        x = self.block(x, training=training) + self.residual_connection(x, training=training)
        x = self.relu(x)

        return x

class AnyHead(layers.Layer):
    """ AnyNet head """

    def __init__(self, num_classes, name='head'):
        super(AnyHead, self).__init__(name=name)

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.0, 0.01))

    def call(self, x, training=True):
        x = self.avg_pool(x, training=training)
        x = self.classifier(x, training=training)

        return x

class AnyStage(layers.Layer):
    """ AnyNet stage (sequence of blocks w/ the same output shape) """

    def __init__(self, depths, in_widths, out_widths, stride, bottleneck_ratio, num_groups, se_ratio, block, name='stage'):
        super(AnyStage, self).__init__(name=name)

        self.blocks = tf.keras.Sequential()
        for d in range(depths):
            block_stride = stride if d == 0 else 1
            projection = True if d == 0 and (in_widths != out_widths or block_stride != 1) else False
            widths = in_widths if d == 0 else out_widths
            self.blocks.add(block(widths, out_widths, block_stride, projection, bottleneck_ratio, num_groups, se_ratio, 'block_{}'.format(d)))

    def call(self, x, training=True):
        return self.blocks(x, training=training)

class AnyNet(tf.keras.Model):
    """ AnyNet model """

    def __init__(self, config, name='model'):
        super(AnyNet, self).__init__(name=name)

        assert len(config.ANYNET.DEPTHS) == len(config.ANYNET.WIDTHS), 'Depths and widths must be specified for each stage.'
        assert len(config.ANYNET.DEPTHS) == len(config.ANYNET.STRIDES), 'Depths and strides must be specified for each stage.'

        BN_PARAMS['momentum'] = config.BN.MOMENTUM
        BN_PARAMS['epsilon'] = config.BN.EPSILON

        # stem
        self.stem = _GetStem(config.ANYNET.STEM_TYPE)(config.ANYNET.STEM_WIDTHS)

        # stages
        block = _GetBlock(config.ANYNET.BLOCK_TYPE)
        if config.ANYNET.BOTTLENECK_RATIOS is None:
            brs = [None for _ in config.ANYNET.WIDTHS]
        elif len(config.ANYNET.BOTTLENECK_RATIOS) == 0:
            brs = [1.0 for _ in config.ANYNET.WIDTHS]
        else:
            brs = config.ANYNET.BOTTLENECK_RATIOS
        if config.ANYNET.NUM_GROUPS is None:
            ngs = [None for _ in config.ANYNET.WIDTHS]
        elif len(config.ANYNET.NUM_GROUPS) == 0:
            ngs = [1 for _ in config.ANYNET.WIDTHS]
        else:
            ngs = config.ANYNET.NUM_GROUPS
        if config.ANYNET.SE_RATIO is None:
            sers = [None for _ in config.ANYNET.WIDTHS]
        else:
            sers = [config.ANYNET.SE_RATIO for _ in config.ANYNET.WIDTHS]
        stage_params = list(zip(config.ANYNET.DEPTHS, config.ANYNET.WIDTHS, config.ANYNET.STRIDES, brs, ngs, sers))

        self.stages = tf.keras.Sequential()
        prev_widths = config.ANYNET.STEM_WIDTHS
        for i, (depths, widths, stride, bottleneck_ratio, num_groups, se_ratio) in enumerate(stage_params):
            self.stages.add(AnyStage(depths, prev_widths, widths, stride, bottleneck_ratio, num_groups, se_ratio, block, 'stage_{}'.format(i)))
            prev_widths = widths

        # head
        self.head = AnyHead(config.MODEL.NUM_CLASSES)

    def call(self, x, training=True):
        x = self.stem(x, training=training)
        x = self.stages(x, training=training)
        x = self.head(x, training=training)

        return x

if __name__ == '__main__':
    config = _BuildConfig()
    model = AnyNet(config)
    x = tf.random.normal((64, 224, 224, 3))
    y = model(x)

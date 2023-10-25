import tensorflow as tf
from keras import Model, Sequential


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor

    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  strides=stride, padding='SAME', use_bias=False, name='Conv2d')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
        self.activation = tf.keras.layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        if expand_ratio != 1:
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))
        layer_list.extend([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                   use_bias=False, name='depthwise'),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
            tf.keras.layers.ReLU(max_value=6.0),
            tf.keras.layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding='SAME', use_bias=False, name='project'),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='project/BatchNorm')
        ])
        self.main_branch = Sequential(layer_list, name='expanded_conv')

    def call(self, inputs, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs)
        else:
            return self.main_branch(inputs)


def mobilenet_v2_5(im_height=224, im_width=224, num_classes=1000, alpha=1.0, round_nearest=8):
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    input_image = tf.keras.layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            x = block(x.shape[-1], output_channel, stride, expand_ratio=t)(x)
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, name='Logits')(x)

    model = Model(inputs=input_image, outputs=output)
    return model

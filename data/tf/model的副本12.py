from tensorflow.keras import layers, Model, Sequential


def _make_divisible(ch, divisor=8, min_ch=None):
    # ch参数是指通过alpha调整之后的卷积核的个数
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    # 下面的这个运算是为了将卷积核的个数圆整到离他最近的8的倍数
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  strides=stride, padding='SAME', use_bias=False, name='Conv2d')
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
        self.activation = layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        # 这里的expand_ratio就是倒残差结构中间的通道数相比于输入通道数的比例
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        # 判断倒残差模块的扩展因子是不是为1，如果是1，那么倒残差模块中间的通道数不增加，于是第一个1*1卷积就没有起到增加通道数的作用，我们便不加1*1的卷积，这是参考了pytorch和tensorflow的官方实现
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))
        layer_list.extend([
            # 3x3 depthwise conv
            layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                   use_bias=False, name='depthwise'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
            layers.ReLU(max_value=6.0),
            # 1x1 pointwise conv(linear)
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding='SAME', use_bias=False, name='project'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='project/BatchNorm')
        ])
        self.main_branch = Sequential(layer_list, name='expanded_conv')

    def call(self, inputs, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs)
        else:
            return self.main_branch(inputs)


def MobileNetV2(im_height=224, im_width=224, num_classes=1000, alpha=1.0, round_nearest=8):
    """
    alpha：就是控制卷积核个数的超参数，目的是为了压缩模型
    round_nearest：是指将卷积核的个数都圆整为8的整数倍
    """
    block = InvertedResidual
    # input_channel是指网络第一层卷积核的个数，last_channel是指网络GlobalAvgPool之前的最后一个卷积层的kernel数量
    # _make_divisible函数保证了通过alpha超参数调整后的卷积核的个数圆整到距离他最近的8的倍数
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    # 倒残差结构参数配置中的s是指倒残差模块中 第一个 倒残差结构的stride，而且这个stride只决定3*3s深度可分离卷积的stride，1*1卷积的stride始终是1。同一模块中的其他倒残差结构的stride都是1。
    # t扩展因子决定了残差结构中第一层1*1卷积核的个数。
    # c通道数是一个残差结构的输出通道数，他决定了残差结构中最后一层1*1卷积核的个数
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

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # conv1
    x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    # building inverted residual blockes
    for t, c, n, s in inverted_residual_setting:
        # 用alpha超参数调整倒残差结构的输出通道
        output_channel = _make_divisible(c * alpha, round_nearest)
        # 一个模块中有n个倒残差结构
        for i in range(n):
            # 每个模块中的第一个倒残差结构的stride可变，后面的倒残差结构的stride都是1
            stride = s if i == 0 else 1
            x = block(x.shape[-1], output_channel, stride, expand_ratio=t)(x)
    # building last several layers
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)

    # building classifier
    x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, name='Logits')(x)
    # 这里的网络最后没有用softmax，就是因为softmax会造成数值计算不稳定，从而使得模型的性能变差

    model = Model(inputs=input_image, outputs=output)
    return model

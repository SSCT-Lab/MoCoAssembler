import paddle
from paddle import nn


class BNConvLayer(nn.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 filter_size,
                 stride=1,
                 groups=1,
                 act='relu'):
        super(BNConvLayer, self).__init__(name_scope)

        self._conv = nn.Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 批规范化
        self._batch_norm = nn.BatchNorm(self.full_name(), num_channels, act=act)

    def forward(self, inputs):
        y = self._batch_norm(inputs)
        y = self._conv(y)

        return y


class BottleneckLayer(nn.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(BottleneckLayer, self).__init__(name_scope)

        self.bn_conv1 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters * 4,
            num_channels=num_channels,
            filter_size=1)

        self.bn_conv2 = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_filters * 4,
            filter_size=3)
        self.drop_out_prob = drop_out_prob

    def forward(self, inputs):
        y = self.bn_conv1(inputs)
        y = nn.Dropout(x=y, dropout_prob=self.drop_out_prob)
        y = self.bn_conv2(y)
        y = nn.Dropout(x=y, dropout_prob=self.drop_out_prob)
        return y


class DenseBlock(nn.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(DenseBlock, self).__init__(name_scope)

        self.block = BottleneckLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            drop_out_prob=drop_out_prob)

        self.block_num = block_num
        channels = num_channels + num_filters
        self.convs = []

        for i in range(self.block_num - 1):
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                BottleneckLayer(
                    self.full_name(),
                    num_filters=num_filters,
                    num_channels=channels,
                    drop_out_prob=drop_out_prob))
            self.convs.append(conv_block)
            channels = channels + num_filters

        self.out_channel = channels

    def forward(self, inputs):
        layers = []
        layers.append(inputs)
        y = self.block(inputs)
        layers.append(y)
        for conv in self.convs:
            y = paddle.concat(layers, axis=1)
            y = conv(y)
            layers.append(y)
        y = paddle.concat(layers, axis=1)
        return y


class TransitionLayer(nn.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 drop_out_prob):
        super(TransitionLayer, self).__init__(name_scope)

        # Bottleneck Layer
        # 将上一步dense block 的输出用1*1卷积降维，输出channel为k(=growth rate)个
        self.conv = BNConvLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            filter_size=1)

        # 池化层
        self.pool2d = nn.Pool2D(
            self.full_name(),
            pool_size=2,
            pool_stride=2,
            pool_type='avg')

        self.dropout_prob = drop_out_prob

    def forward(self, inputs):
        y = self.conv(inputs)
        y = nn.Dropout(x=y, dropout_prob=self.dropout_prob)
        y = self.pool2d(y)

        return y


class LoopLayer(nn.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 num_channels,
                 block_num,
                 drop_out_prob):
        super(LoopLayer, self).__init__(name_scope)

        # 加入一个dense block
        self.denseblock = DenseBlock(
            self.full_name(),
            num_filters=num_filters,
            num_channels=num_channels,
            block_num=block_num,
            drop_out_prob=drop_out_prob)

        self.channel = self.denseblock.out_channel

        # 加入一个translation layer
        self.transblock = TransitionLayer(
            self.full_name(),
            num_filters=num_filters,
            num_channels=self.channel,
            drop_out_prob=drop_out_prob)

    def forward(self, inputs):
        y = self.denseblock(inputs)
        y = self.transblock(y)
        return y


class DenseNet(nn.Layer):
    def __init__(self, name_scope, layers, dropout_prob, class_dim=5):
        super(DenseNet, self).__init__(name_scope)

        self.layers = layers
        self.dropout_prob = dropout_prob

        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        self.conv1 = nn.Conv2D(
            self.full_name(),
            num_filters=layer_conf[0] * 2,  # 使用2k个filter
            filter_size=7,  # 7*7卷积
            stride=2,  # 步长为2，每进行一次卷积后向后移动两个单位
            padding=3,  # 在周围补齐宽度为3的0，使得可以对每一个位置进行卷积
            groups=1,
            act=None,
            bias_attr=False)

        self.pool1 = nn.Pool2D(
            self.full_name(),
            pool_size=3,
            pool_padding=1,
            pool_stride=2,
            pool_type='max')
        channels = layer_conf[0] * 2

        self.convs = []
        for i in range(len(layer_conf[1]) - 1):
            conv_block = self.add_sublayer(
                'bb_%d_%d' % (i, i),
                LoopLayer(
                    self.full_name(),
                    num_filters=layer_conf[0],
                    num_channels=channels,
                    block_num=layer_conf[1][i],
                    drop_out_prob=self.dropout_prob
                ))
            channels = layer_conf[0]
            self.convs.append(conv_block)

        self.conv3 = DenseBlock(
            self.full_name(),
            num_filters=layer_conf[1][-1],
            num_channels=layer_conf[0],
            block_num=layer_conf[0],
            drop_out_prob=self.dropout_prob)

        self.pool2 = Pool2D(
            self.full_name(),
            global_pooling=True,
            pool_type='avg')

        self.fc = FC(self.full_name(),
                     size=class_dim,
                     act='softmax')

    def forward(self, inputs, label=None):
        y = self.conv1(inputs)
        y = self.pool1(y)
        for conv in self.convs:
            y = conv(y)
        y = self.conv3(y)
        y = self.pool2(y)
        y = self.fc(y)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y

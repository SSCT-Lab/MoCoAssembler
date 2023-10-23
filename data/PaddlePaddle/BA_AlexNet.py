import paddle


class BA_module(paddle.nn.Layer):
    def __init__(self, pre_channels, cur_channel, reduction=16):
        super().__init__()
        self.pre_fusions = paddle.nn.LayerList(
            [paddle.nn.Sequential(
                paddle.nn.AdaptiveAvgPool2D(1),
                paddle.nn.Conv2D(pre_channel, cur_channel // reduction, 1, bias_attr=False),
                paddle.nn.BatchNorm2D(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = paddle.nn.Sequential(
            paddle.nn.AdaptiveAvgPool2D(1),
            paddle.nn.Conv2D(cur_channel, cur_channel // reduction, 1, bias_attr=False),
            paddle.nn.BatchNorm2D(cur_channel // reduction)
        )

        self.generation = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(cur_channel // reduction, cur_channel, 1, bias_attr=False),
            paddle.nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.shape

        pre_fusions = [self.pre_fusions[i](pre_layers[i]) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer)
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion)

        return att_weights

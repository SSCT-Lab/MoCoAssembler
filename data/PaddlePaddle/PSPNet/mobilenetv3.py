import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import numpy as np

__all__ = ['MobileNetV3', 'mobilenetv3']


class ReLU(dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        return fluid.layers.relu(inputs)

def conv_bn(inp, oup, stride, conv_layer=dygraph.Conv2D, norm_layer=dygraph.BatchNorm, nlin_layer=ReLU):
    return dygraph.Sequential(
        conv_layer(inp, oup, 3, stride, 1),
        norm_layer(oup),
        nlin_layer()
    )


def conv_1x1_bn(inp, oup, conv_layer=dygraph.Conv2D, norm_layer=dygraph.BatchNorm, nlin_layer=ReLU):
    return dygraph.Sequential(
        conv_layer(inp, oup, 1, 1, 0),
        norm_layer(oup),
        nlin_layer()
    )


class Hswish(dygraph.Layer):
    def __init__(self):
        super(Hswish, self).__init__()

    def forward(self, x):
        return x * fluid.layers.relu6(x + 3.) / 6.


class Hsigmoid(dygraph.Layer):
    def __init__(self):
        super(Hsigmoid, self).__init__()

    def forward(self, x):
        return fluid.layers.relu6(x + 3.) / 6.


class SEModule(dygraph.Layer):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = dygraph.Pool2D(pool_type='avg',global_pooling=True)
        self.fc = dygraph.Sequential(
            dygraph.Linear(channel, channel // reduction),
            ReLU(),
            dygraph.Linear(channel // reduction, channel),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = fluid.layers.reshape(y, [b, c])
        y = self.fc(y)
        y = fluid.layers.reshape(y, [b, c, 1, 1])
        y = fluid.layers.expand_as(y, target_tensor=x)
        result = fluid.layers.elementwise_mul(x, y)
        return result


class Identity(dygraph.Layer):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(dygraph.Layer):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = dygraph.Conv2D
        norm_layer = dygraph.BatchNorm
        if nl == 'RE':
            nlin_layer = ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = dygraph.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0),
            norm_layer(exp),
            nlin_layer(),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(dygraph.Layer):
    def __init__(self,au = 0, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        self.au = au
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                # [5, 672, 160, True,  'HS', 2],
                # [5, 960, 160, True,  'HS', 1],
                # [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                # [5, 288, 96,  True,  'HS', 2],
                # [5, 576, 96,  True,  'HS', 1],
                # [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.low_features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.atrous_features = []
        self.classifier = []
        stride = 2
        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            stride = stride * 2
            if stride <= 4:
                self.low_features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            else :
                self.atrous_features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        # if mode == 'large':
        #     last_conv = make_divisible(960 * width_mult)
        #     self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     self.features.append(dygraph.Pool2D(pool_type='avg', global_pooling=True))
        #     self.features.append(dygraph.Conv2D(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish())
        # elif mode == 'small':
        #     last_conv = make_divisible(576 * width_mult)
        #     self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        #     self.features.append(dygraph.Pool2D(pool_type='avg', global_pooling=True))
        #     self.features.append(dygraph.Conv2D(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish())
        # else:
        #     raise NotImplementedError

        # make it dygraph.Sequential
        self.low_features = dygraph.Sequential(*self.low_features)
        self.atrous_features = dygraph.Sequential(*self.atrous_features)

        # building classifier
        # self.classifier = dygraph.Sequential(
        #     # dygraph.Dropout(dropout),    # refer to paper section 6
        #     dygraph.Linear(last_channel, n_class)
        # )


    def forward(self, x):
        low_features = self.low_features(x)
        
        atrous_features = self.atrous_features(low_features)
        # x = x.mean(3).mean(2)
        if not self.au:
            return atrous_features
        else:
            return atrous_features, low_features

def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        # state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        # model.load_state_dict(state_dict, strict=True)
        raise NotImplementedError
    return model


if __name__ == '__main__':
    with fluid.dygraph.guard():
        batch_size = 1
        in_dims = 3
        num_class = 8
        im = np.random.randint(0, 256, (4, 3, 448, 448)).astype('float32')
        im = dygraph.to_variable(im)
        print(im.shape)

        model = mobilenetv3()
        output = model(im)
        print('atrous shape:{}'.format(output[0].shape))
        print('low level shape:{}'.format(output[1].shape))
        pass

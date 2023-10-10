
import paddle
import paddle.nn as nn

import numpy as np

import math
from yacs.config import CfgNode as CN

#####################################################################3
# part 1: functions
#####################################################################3

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_tanh(nn.Layer):
    def __init__(self, h_max=1):
        super(h_tanh, self).__init__()
        self.relu = nn.ReLU6()
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3)*self.h_max / 3 - self.h_max

class h_swish(nn.Layer):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def conv_3x3_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1, bias_attr=False, dilation=dilation),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU6()
    )

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=False),
        nn.BatchNorm2D(inp),
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU(),
    )
def gcd(a, b):
    a, b = (a, b) if a >= b else (b, a)
    while b:
        a, b = b, a%b
    return a

def get_act_layer(inp, oup, mode='SE1', act_relu=True, act_max=2, act_bias=True, init_a=[1.0, 0.0], reduction=4, init_b=[0.0, 0.0], g=None, act='relu', expansion=True):
    layer = None
    if mode == 'SE1':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
            nn.ReLU6() if act_relu else nn.Sequential()
        )
    elif mode == 'SE0':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction), 
        )
    elif mode == 'NA':
        layer = nn.ReLU6() if act_relu else nn.Sequential()
    elif mode == 'LeakyReLU':
        layer = nn.LeakyReLU() if act_relu else nn.Sequential()
    elif mode == 'RReLU':
        layer = nn.RReLU() if act_relu else nn.Sequential()
    elif mode == 'PReLU':
        layer = nn.PReLU() if act_relu else nn.Sequential()
    elif mode == 'DYShiftMax':
        layer = DYShiftMax(inp, oup, act_max=act_max, act_relu=act_relu, init_a=init_a, reduction=reduction, init_b=init_b, g=g, expansion=expansion)
    return layer

def get_squeeze_channels(inp, reduction):
    if reduction == 4:
        squeeze = inp // reduction
    else:
        squeeze = _make_divisible(inp // reduction, 4)
    return squeeze
#####################################################################3
# part 2: modules
#####################################################################3
class SELayer(nn.Layer):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.oup = oup
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        # determine squeeze
        squeeze = get_squeeze_channels(inp, reduction)
        print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))


        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(),
                nn.Linear(squeeze, oup),
                h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).reshape((b, c))
        y = self.fc(y).reshape((b, self.oup, 1, 1))
        return x_out * y

class DYShiftMax(nn.Layer):
    def __init__(self, inp, oup, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=None, expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU() if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2D(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4
        print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        print('init-a: {}, init-b: {}'.format(init_a, init_b))

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        print('group shuffle: {}, divide group: {}'.format(self.g, expansion))
        self.gc = inp//self.g

        index=paddle.Tensor(np.array(list(range(inp)))).reshape((1,inp,1,1))
        index=index.reshape((1,self.g,self.gc,1,1))
        indexgs = paddle.split(index, [1, self.g-1], axis=1)
        indexgs = paddle.concat((indexgs[1], indexgs[0]), axis=1)
        indexs = paddle.split(indexgs, [1, self.gc-1], axis=2)
        indexs = paddle.concat((indexs[1], indexs[0]), axis=2)
        self.index = indexs.reshape(inp)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.shape
        y = self.avg_pool(x_in).reshape((b, c))
        y = self.fc(y).reshape((b, self.oup*self.exp, 1, 1))
        y = (y-0.5) * self.act_max

        n2, c2, h2, w2 = x_out.shape
        x2 = x_out[:,self.index,:,:]

        if self.exp == 4:
            a1, b1, a2, b2 = paddle.split(y, self.oup, axis=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = paddle.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = paddle.split(y, self.oup, axis=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out

class MaxGroupPooling(nn.Layer):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.size()

        # reshape
        y = x.reshape((b, c // self.channel_per_group, -1, h, w))
        out, _ = paddle.max(y, axis=2)
        return out

class SwishLinear(nn.Layer):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)

class StemLayer(nn.Layer):
    def __init__(self, inp, oup, stride, dilation=1, mode='default', groups=(4,4)):
        super(StemLayer, self).__init__()

        self.exp = 1 if mode == 'default' else 2
        g1, g2 = groups 
        if mode == 'default':
            self.stem = nn.Sequential(
                nn.Conv2D(inp, oup*self.exp, 3, stride, 1, bias_attr=False, dilation=dilation),
                nn.BatchNorm2D(oup*self.exp),
                nn.ReLU6() if self.exp == 1 else MaxGroupPooling(self.exp)
            )
        elif mode == 'spatialsepsf':
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.ReLU6()
            )
        else: 
            raise ValueError('Undefined stem layer')
           
    def forward(self, x):
        out = self.stem(x)    
        return out

class GroupConv(nn.Layer):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        # print ('inp: %d, oup:%d, g:%d' %(inp, oup, self.groups[0]))
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False, groups=self.groups[0]),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelShuffle(nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.reshape((b, self.groups, channels_per_group, h, w))

        x = x.transpose([0, 2, 1, 3, 4])
        out = x.reshape((b, -1, h, w))

        return out

class ChannelShuffle2(nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle2, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape

        channels_per_group = c // self.groups

        # reshape
        x = x.reshape((b, self.groups, channels_per_group, h, w))
        x = x.transpose([0, 2, 1, 3, 4])
        out = x.reshape((b, -1, h, w))

        return out

def upsample(inp, oup, scale=2):
    return nn.Sequential(
                nn.Conv2D(inp, inp, 3, 1, 1, groups=inp),
                nn.ReLU(),
                conv_1x1_bn(inp,oup),
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))

######################################################################3
# part 3: new block
#####################################################################3

class SpatialSepConvSF(nn.Layer):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias_attr=False, groups=1
            ),
            nn.BatchNorm2D(oup1),
            nn.Conv2D(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias_attr=False, groups=oup1
            ),
            nn.BatchNorm2D(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConv(nn.Layer):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(inp, oup, kernel_size, stride, kernel_size//2, bias_attr=False, groups=inp),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Layer):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp*exp1
        oup = inp*exp1*exp2
        
        self.conv = nn.Sequential(
            nn.Conv2D(inp, inp*exp1, 
                (kernel_size, 1), 
                (stride, 1), 
                (kernel_size//2, 0), 
                bias_attr=False, groups=inp
            ),
            nn.BatchNorm2D(inp*exp1),
            nn.Conv2D(hidden_dim, oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias_attr=False, groups=hidden_dim
            ),
            nn.BatchNorm2D(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def get_pointwise_conv(mode, inp, oup, hiddendim, groups):

    if mode == 'group':
        return GroupConv(inp, oup, groups)
    elif mode == '1x1':
        return nn.Sequential(
                    nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
                    nn.BatchNorm2D(oup)
                )
    else:
        return None
 

class DYMicroBlock(nn.Layer):
    def __init__(self, inp, oup, kernel_size=3, stride=1, ch_exp=(2, 2), ch_per_group=4, groups_1x1=(1, 1), depthsep=True, shuffle=False, pointwise='fft', activation_cfg=None):
        super(DYMicroBlock, self).__init__()


        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg.dy
        act = "PReLU"
        act_max = 1.0
        act_bias = True
        act_reduction = 4 * activation_cfg.ratio
        init_a = [1.0, 0.0]
        init_b = [0.0, 0.0]
        init_ab3 = [1.0, 0.0]

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle2(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle2(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle2(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)), #FFTConv
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out

###########################################################################

class Backbone(nn.Layer):
    def __init__(self, input_size=224, num_classes=1000, teacher=False):
        super(Backbone, self).__init__()

        self.cfgs = [
                #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
                [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
                [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
                [1, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
                [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
                [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
                [1, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
                [2, 1,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
            ]

        block = eval("DYMicroBlock")
        stem_mode = "default"
        stem_ch = 16
        stem_dilation = 1
        stem_groups = [4, 8]
        out_ch = 1024
        depthsep = True
        shuffle = False
        pointwise = 'group'
        dropout_rate = 0.0

        act_max = 1.0
        act_bias = True
        activation_cfg = CN()

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation, 
                    mode=stem_mode,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)


        self.
        
        self._initialize_weights()
           
    def forward(self, x):
        x = self.features(x)
        
        #x = self.avgpool(x)
        return x

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # print(dir(m.weight))
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                attr_p = paddle.ParamAttr(initializer=nn.initializer.Normal(0, math.sqrt(2. / n)))
                m.weight = paddle.create_parameter(m.weight.shape, dtype=m.weight.dtype, attr= attr_p)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            elif isinstance(m, nn.Linear):
                attr_p = paddle.ParamAttr(initializer=nn.initializer.Normal(0, 0.01))
                m.weight = paddle.create_parameter(m.weight.shape, dtype=m.weight.dtype, attr= attr_p)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))


class Header(nn.Layer):
    def __init__(self, num_classes, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        #heatmaps, centers, regs, offsets
        #Person keypoint heatmap
        self.header_heatmaps = nn.Sequential(*[
                        dw_conv3(96, 96),
                        nn.Conv2D(96, num_classes, 1, 1, 0,  ),
                        nn.Sigmoid()
                    ])

        #Person center heatmap
        self.header_centers = nn.Sequential(*[
                        dw_conv3(96, 96),
                        nn.Conv2D(96, 1, 1, 1, 0,  ),
                        nn.Sigmoid(),
                        # MulReshapeArgMax()
                    ])

        #Keypoint regression field:
        self.header_regs = nn.Sequential(*[
                        dw_conv3(96, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0,  ),
                    ])

        #2D per-keypoint offset field
        self.header_offsets = nn.Sequential(*[
                        dw_conv3(96, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0,  ),
                    ])

    def argmax2loc(self, x, h=48, w=48):
        ## n,1
        y0 = paddle.div(x,w).long()
        x0 = paddle.sub(x, y0*w).long()
        return x0,y0


    def forward(self, x):

        res = []
        if self.mode=='train':
            h1 = self.header_heatmaps(x)
            h2 = self.header_centers(x)
            h3 = self.header_regs(x)
            h4 = self.header_offsets(x)
            res = [h1,h2,h3,h4]


        elif self.mode=='test':
            pass

        elif self.mode=='all':
            pass
        else:
            print("[ERROR] wrong mode.")

        

        return res

class MoveNet(nn.Layer):
    def __init__(self, num_classes=17, width_mult=1.,mode='train'):
        super(MoveNet, self).__init__()

        self.backbone = Backbone()

        self.header = Header(num_classes, mode)
        

        self._initialize_weights()


    def forward(self, x):
        x = self.backbone(x) # n,24,48,48
        print(x.shape)

        x = self.header(x)
        # print([x0.shape for x0 in x])

        return x


    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # m.weight.data.normal_(0, 0.01)
                attr_p = paddle.ParamAttr(initializer=nn.initializer.KaimingNormal())
                m.weight = paddle.create_parameter(m.weight.shape, dtype=m.weight.dtype, attr= attr_p)
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()




if __name__ == "__main__":

    model = MoveNet()
    print(paddle.summary(model, (1, 3, 192, 192)))


    dummy_input1 = paddle.randn((1, 3, 192, 192))
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    
    # torch.onnx.export(model, dummy_input1, "pose.onnx", 
    #     verbose=True, input_names=input_names, output_names=output_names,
    #     do_constant_folding=True,opset_version=11)
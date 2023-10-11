import paddle
import paddle.nn as nn
from math import ceil
from paddle.vision.models import resnet50
import pickle
import numpy as np


class ConvBNLayer(nn.Layer):
    def __init__(self, in_channels, channels, kernel=3, stride=1, pad=0, num_group=1, bias=False, act="relu6"):
        super(ConvBNLayer, self).__init__()

        conv_ = None
        if stride == 2:
            conv_ = nn.Conv2D(in_channels, channels, kernel, stride, [1, 0, 1, 0], groups=num_group, bias_attr=bias)
        else:
            conv_ = nn.Conv2D(in_channels, channels, kernel, stride, kernel // 2, groups=num_group, bias_attr=bias)

        bn_ = nn.BatchNorm2D(channels)
        act_ = None
        if act == 'swish':
            act_ = nn.Swish()
        elif act == 'relu':
            act_ = nn.ReLU()
        elif act == 'relu6':
            act_ = nn.ReLU6()

        self.conv_bn = nn.Sequential(
            conv_,
            bn_
        )

        if act_ is not None:
            self.conv_bn = nn.Sequential(
                conv_,
                bn_,
                act_
            )

    def forward(self, inputs):
        return self.conv_bn(inputs)


class HarDBlock(nn.Layer):

    # 获取层连接
    def get_link(self, layer, base_ch, growth_rate, grmul):
        # 检查层
        if layer == 0:
            return base_ch, 0, []

        # 计算输出的通道数
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:  # 间隔2的n次方
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul

        out_channels = int(int(out_channels + 1) / 2) * 2

        # 计算之前层的输出，也就是当前层的输出
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super(HarDBlock, self).__init__()

        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvBNLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.LayerList(layers_)
        # print("layers: ", len(self.layers))

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):

            link = self.links[layer]
            # print("HarDBlock layer: ", layer, link)
            tin = []
            for i in link:
                tin.append(layers_[i])

            if len(tin) > 1:
                x = paddle.concat(x=tin, axis=1)
                # print("===>concat: ", x.shape)
            else:
                x = tin[0]
            # print(self.layers[layer])
            out = self.layers[layer](x)
            # print(x.shape, out.shape)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])

        out = paddle.concat(x=out_, axis=1)
        return out


class HarDNet68(nn.Layer):
    def __init__(self, cls_num=1000):
        super(HarDNet68, self).__init__()

        # 模型的head
        base = []
        base.append(ConvBNLayer(3, 32, kernel=3, stride=2, bias=False))
        base.append(ConvBNLayer(32, 64, kernel=3))
        base.append(nn.MaxPool2D(kernel_size=3, stride=2, padding=1))

        # 构建HarDBlock
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]
        grmul = 1.7
        drop_rate = 0.1
        blks = len(n_layers)

        ch = 64
        for i in range(blks):

            # blk = self.add_sublayer("HarDBlock_" + str(i), HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=False))
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=False)

            ch = blk.get_out_ch()
            base.append(blk)

            # print("fucking...===>", ch, ch_list[i])
            base.append(ConvBNLayer(ch, ch_list[i], kernel=1))
            # print(self.base[-1])

            ch = ch_list[i]
            if downSamp[i] == 1:
                base.append(nn.MaxPool2D(kernel_size=2, stride=2))

        ch = ch_list[blks - 1]
        base.append(nn.AdaptiveAvgPool2D(output_size=1))
        base.append(nn.Flatten())
        base.append(nn.Dropout(drop_rate))
        base.append(nn.Linear(ch, cls_num))

        self.base = nn.Sequential(*base)

    def forward(self, x):
        for i, layer in enumerate(self.base):
            x = layer(x)
        return x
import paddle.fluid.layers as F
import paddle.fluid.dygraph as nn
import numpy as np


class ReLU(nn.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return F.relu(x)


class ConvBnReLU(nn.Layer):
    def __init__(self, in_ch, out_ch, filter_size=3, stride=1, padding=1, dilation=1, bias_attr=False):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2D(int(in_ch), int(out_ch), filter_size, stride, int(padding * dilation),
                              dilation=int(dilation), bias_attr=bias_attr)
        self.bn = nn.BatchNorm(int(out_ch))
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


'''RSU 模块'''


class RSU(nn.Layer):
    def __init__(self, layer, in_ch, mid_ch, out_ch):
        super(RSU, self).__init__()
        self.stem = ConvBnReLU(in_ch, out_ch)
        self.pool = nn.Pool2D(2, "max", 2)
        self.layer = layer
        '''下采样'''
        self.conv = [ConvBnReLU(out_ch, mid_ch)] + [ConvBnReLU(mid_ch, mid_ch) for _ in range(layer - 1)]

        '''上采样'''
        self.convd = [ConvBnReLU(mid_ch * 2, mid_ch) for _ in range(layer - 1)] + [ConvBnReLU(mid_ch * 2, out_ch)]

        '''中间'''
        self.mid_conv = ConvBnReLU(mid_ch, mid_ch, dilation=2)

        self.conv = nn.LayerList(self.conv)
        self.convd = nn.LayerList(self.convd)

    def forward(self, x):
        tmp = hxin = self.stem(x)
        hx = []

        '''下采样(最后一层不需要)'''
        for i in range(self.layer):
            tmp = self.conv[i](tmp)
            hx.append(tmp)
            if i != self.layer - 1:
                tmp = self.pool(tmp)

        tmp = self.mid_conv(hx[-1])

        '''上采样 第一层不需要'''
        for i in range(self.layer):
            tmp = self.convd[i](F.concat([tmp, hx[-(i + 1)]], axis=1))
            if i != self.layer - 1:
                tmp = F.interpolate(tmp, scale=2)
        return tmp + hxin


class RSUF(nn.Layer):
    def __init__(self, layer, in_ch, mid_ch, out_ch):
        super(RSUF, self).__init__()
        self.layer = layer

        self.stem = ConvBnReLU(in_ch, out_ch)

        dilation = 1
        self.conv = [ConvBnReLU(out_ch, mid_ch)]
        for i in range(self.layer - 1):
            dilation *= 2
            self.conv.append(ConvBnReLU(mid_ch, mid_ch, dilation=dilation))

        self.mid_conv = ConvBnReLU(mid_ch, mid_ch, dilation=dilation * 2)

        self.convd = []
        dilation = 8
        for i in range(self.layer - 1):
            dilation /= 2
            self.convd.append(ConvBnReLU(mid_ch * 2, mid_ch, dilation=dilation))
        self.convd.append(ConvBnReLU(mid_ch * 2, out_ch))

        self.conv = nn.LayerList(self.conv)
        self.convd = nn.LayerList(self.convd)

    def forward(self, x):
        tmp = hxin = self.stem(x)

        hx = []
        for i in range(self.layer):
            tmp = self.conv[i](tmp)
            hx.append(tmp)

        tmp = self.mid_conv(hx[-1])

        for i in range(self.layer):
            tmp = self.convd[i](F.concat([tmp, hx[-(i + 1)]], axis=1))

        return tmp + hxin


class U2Net(nn.Layer):
    def __init__(self, in_ch=3, num_class=3):
        super(U2Net, self).__init__()
        self.pool = nn.Pool2D(2, "max", 2)
        self.layer = 6

        self.stage = [RSU(self.layer, in_ch, 32, 64)]
        in_ch, mid_ch, out_ch = 64, 32, 128

        for i in range(self.layer - 3):
            self.stage.append(RSU(self.layer - i - 1, in_ch, mid_ch, out_ch))
            in_ch, mid_ch, out_ch = 2 * in_ch, 2 * mid_ch, 2 * out_ch
        self.stage.append(RSUF(3, 512, 256, 512))

        self.mid_stage = RSUF(3, 512, 256, 512)

        in_ch, mid_ch, out_ch = 1024, 256, 512
        self.staged = [RSUF(3, in_ch, mid_ch, out_ch)]
        mid_ch, out_ch = mid_ch / 2, out_ch / 2

        for i in range(self.layer - 3):
            self.staged.append(RSU(i + self.layer - 3, in_ch, mid_ch, out_ch))
            in_ch, mid_ch, out_ch = in_ch / 2, mid_ch / 2, out_ch / 2
        self.staged.append(RSU(self.layer, in_ch, mid_ch, 2 * out_ch))

        out_ch = out_ch * 2
        self.side = [nn.Conv2D(int(out_ch), num_class, 3, padding=1, bias_attr=True)]
        for i in range(self.layer - 2):
            self.side.append(nn.Conv2D(int(out_ch), num_class, 3, padding=1, bias_attr=True))
            out_ch *= 2
        self.side.append(nn.Conv2D(int(out_ch / 2), num_class, 3, padding=1, bias_attr=True))
        self.side.reverse()

        self.out_conv = nn.Conv2D(self.layer * num_class, num_class, 1)

        self.side = nn.LayerList(self.side)
        self.stage = nn.LayerList(self.stage)
        self.staged = nn.LayerList(self.staged)

    def forward(self, x):
        or_shape = x.shape[2:]
        hx = []
        for i in range(self.layer - 1):
            x = self.stage[i](x)
            hx.append(x)
            x = self.pool(x)

        hxd = []
        x = self.mid_stage(x)

        for i in range(self.layer - 1):
            hxd.append(x)
            x = F.interpolate(x, scale=2)
            x = self.staged[i](F.concat([x, hx[-(i + 1)]], axis=1))
        hxd.append(x)

        sides = [self.side[i](hxd[i]) for i in range(self.layer)]
        sides = [F.interpolate(side, out_shape=or_shape) for side in sides]
        sides.append(self.out_conv(F.concat(sides, axis=1)))
        sides = [F.softmax(side, axis=1) for side in sides]
        return reversed(sides)


if __name__ == '__main__':
    with nn.guard():
        model = U2Net()
        model.eval()
        img = np.zeros([1, 3, 512, 512]).astype('float32')
        img = nn.to_variable(img)
        outs = model(img)
        for out in outs:
            print(out.shape)

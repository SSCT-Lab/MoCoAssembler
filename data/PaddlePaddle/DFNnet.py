import paddle.fluid.dygraph as nn
import paddle.fluid as fluid


class ReLU(nn.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x = fluid.layers.relu(x)
        return x


class Conv2DTranspose(nn.Layer):
    def __init__(self, in_channels, out_channels, size=None, factor=None):
        super(Conv2DTranspose, self).__init__()
        self.factor = factor
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2D(self.in_channels, self.out_channels, 3, 1, 1)

    def forward(self, x):
        x = fluid.layers.image_resize(x, self.size, self.factor)
        x = self.conv(x)
        return x


class Bottleneck(nn.Layer):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(inplanes, planes, 1, bias_attr=False),
            nn.BatchNorm(planes),
            ReLU(),

            nn.Conv2D(planes, planes, 3, stride=stride, padding=1, bias_attr=False),
            nn.BatchNorm(planes),
            ReLU(),

            nn.Conv2D(planes, planes * 4, 1, bias_attr=False),
            nn.BatchNorm(planes * 4))

        self.relu = ReLU()

        self.downsample = nn.Sequential(
            nn.Conv2D(inplanes, planes * 4, 1, stride=stride),
            nn.BatchNorm(planes * 4))
        self.ds = (inplanes != planes * 4)

    def forward(self, x):
        out = self.conv(x)
        if self.ds:
            x = self.downsample(x)
        out += x
        return self.relu(out)


class ResNet(nn.Layer):
    def __init__(self, layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        planes = [64, 128, 256, 512]
        inplanes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, inplanes, 7, stride=2, padding=3, bias_attr=False),
            nn.BatchNorm(inplanes),
            ReLU(),
            nn.Pool2D(pool_size=3, pool_type="max", pool_stride=2, pool_padding=1)
        )
        for i in range(4):
            block = nn.Sequential(Bottleneck(inplanes, planes[i], stride=(i + 7) // 4))
            inplanes = planes[i] * 4
            for j in range(1, layers[i]):
                block.add_sublayer(str(j), Bottleneck(inplanes, planes[i]))
            self.add_sublayer('conv' + str(i + 2), block)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        global_pool = fluid.layers.adaptive_pool2d(out5, (1, 1), "avg")
        return out2, out3, out4, out5, global_pool


class RRB(nn.Layer):
    def __init__(self, inplanes, planes, interplanes=512):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2D(planes, interplanes, 3, padding=1),
            nn.BatchNorm(planes),
            ReLU(),
            nn.Conv2D(interplanes, planes, 3, padding=1))
        self.relu = ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        return self.relu(out + out1)


class CAB(nn.Layer):
    def __init__(self, inplanes, interplanes=512):
        super(CAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(inplanes * 2, interplanes, 1),
            ReLU(),
            nn.Conv2D(interplanes, inplanes, 1))

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            print(x1.shape, x2.shape)
        x = fluid.layers.concat([x1, x2], 1)
        out = fluid.layers.adaptive_pool2d(x, 1, "avg")
        out = self.conv1(out)
        out = fluid.layers.sigmoid(out)
        return out * x1 + x2


def side_branch(class_num, factor, inplanes=512):
    branch = nn.Sequential(
        nn.Conv2D(inplanes, class_num, 1),
        # hout = (hin - kernel + 2*padding)/stride +1 :(32-1+0)/1+1=32  hout = (hin-1)*stride-2*padding+kernel+outputpadding
        Conv2DTranspose(class_num, class_num, factor=factor))  # (512-1)*2-2*0+2
    return branch


class SmoothBlock(nn.Layer):
    def __init__(self, plane, factor=None, inplanes=512, class_num=0):
        super(SmoothBlock, self).__init__()
        self.factor = factor
        self.rrb = RRB(plane, 512)
        self.cab = CAB(512)
        self.rrbb = RRB(512, 512)
        self.transpose = Conv2DTranspose(512, 512, factor=2)
        if self.factor:
            self.branch = side_branch(class_num, factor, inplanes)

    def forward(self, x1, x2):
        b = None
        input1 = self.rrb(x1)
        # print("before",x2.shape)
        input2 = self.transpose(x2)
        # print("after",input2.shape)
        if self.factor:
            b = self.branch(input2)
        cab_out = self.cab(input1, input2)
        rrb_out = self.rrbb(cab_out)
        return b, rrb_out

    def patch(self, trans):
        self.transpose = trans


class SmoothNet(nn.Layer):
    def __init__(self, class_num):
        super(SmoothNet, self).__init__()
        planes = [2048, 1024, 512, 256]
        factor = [16, 8, 4, 2]
        self.class_num = class_num
        self.block4 = SmoothBlock(planes[0])
        self.block3 = SmoothBlock(planes[1], factor[0], inplanes=512, class_num=class_num)
        self.block2 = SmoothBlock(planes[2], factor[1], inplanes=512, class_num=class_num)
        self.block1 = SmoothBlock(planes[3], factor[2], inplanes=512, class_num=class_num)
        self.transpose = Conv2DTranspose(512, 512, factor=2)
        self.branch = side_branch(class_num, factor[3], 512)
        self.conv = nn.Conv2D(class_num * 4, class_num, 1)
        self.patched = False

    def forward(self, x):
        if not self.patched:
            self.patch(tuple(x[-2].shape[2:4]))
        _, out = self.block4(x[-2], x[-1])
        b4, out = self.block3(x[-3], out)
        b3, out = self.block2(x[-4], out)
        b2, out = self.block1(x[-5], out)
        out = self.transpose(out)
        b1 = self.branch(out)
        b = fluid.layers.concat((b1, b2, b3, b4), 1)
        fuse = self.conv(b)
        return b1, b2, b3, b4, fuse

    def patch(self, pic_size):
        # print(pic_size)
        trans = Conv2DTranspose(2048, 512, pic_size)
        self.block4.patch(trans)
        self.patched = True


class BorderBlock(nn.Layer):
    def __init__(self, planes, stride=2):
        super(BorderBlock, self).__init__()
        self.rrb1 = RRB(planes, 512)
        self.transpose = Conv2DTranspose(512, 512, factor=stride)  # (32-1)*4-2*0+1+3
        self.rrb2 = RRB(512, 512)

    def forward(self, x1, x2):
        out = self.rrb1(x1)
        out = self.transpose(out)
        out = out + x2
        out = self.rrb2(out)
        return out


class BorderNet(nn.Layer):
    def __init__(self, class_num):
        super(BorderNet, self).__init__()
        planes = [256, 512, 1024, 2048]
        self.rrb = RRB(planes[0], 512)
        self.block1 = BorderBlock(planes[1], 2)
        self.block2 = BorderBlock(planes[2], 4)
        self.block3 = BorderBlock(planes[3], 8)
        self.transpose = Conv2DTranspose(512, 512, factor=2)
        self.branch = side_branch(class_num, 2, 512)

    def forward(self, x):
        out = self.rrb(x[0])
        out = self.block1(x[1], out)
        out = self.block2(x[2], out)
        out = self.block3(x[3], out)
        out = self.transpose(out)
        return self.branch(out)


class DFN(nn.Layer):
    def __init__(self, class_num):
        super(DFN, self).__init__()
        self.base = ResNet()
        self.smooth = SmoothNet(class_num)
        self.border = BorderNet(class_num)

    def forward(self, x):
        out = self.base(x)
        b1, b2, b3, b4, fuse = self.smooth(out)
        r1 = self.border(out)
        b1 = fluid.layers.softmax(b1, 1)
        b2 = fluid.layers.softmax(b2, 1)
        b3 = fluid.layers.softmax(b3, 1)
        b4 = fluid.layers.softmax(b4, 1)
        fuse = fluid.layers.softmax(fuse, 1)
        r1 = fluid.layers.softmax(r1, 1)

        return b1, b2, b3, b4, fuse, r1
import paddle
from paddle import nn
from paddle.nn import functional as F

class Identify(nn.Layer):
    """占位符-不做任何操作
    """
    def __init__(self):
        super(Identify, self).__init__()
    
    def forward(self, inputs):
        return inputs

class DepthWise_Conv(nn.Layer):
    """通道分离卷积-group==in_channels
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(DepthWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=in_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class PointWise_Conv(nn.Layer):
    """逐点卷积-1x1
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PointWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class Linear_Depth_Separ_Conv(nn.Layer):
    """线性可分离卷积-末尾逐点卷积为线性输出
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Linear_Depth_Separ_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.depthwise_conv = DepthWise_Conv(in_channels=in_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
        self.pointwise_conv = PointWise_Conv(in_channels=in_channels,
                                             out_channels=out_channels)
        
        self.act = act()
        self.depthwise_bn = nn.BatchNorm2D(in_channels)
        self.pointwise_bn = nn.BatchNorm2D(out_channels)
        
    def forward(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.act(self.depthwise_bn(x))
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)  # linear output
        return x


class SEBlock(nn.Layer):
    """SE Attention Block -- by channels
    """
    def __init__(self,
                 in_channels,
                 reduce=4.):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduce = reduce

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1 = PointWise_Conv(in_channels=in_channels,
                                   out_channels=(in_channels // reduce))
        self.conv2 = PointWise_Conv(in_channels=(in_channels // reduce),
                                   out_channels=in_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.Hardsigmoid()

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = inputs * x # origin input fuses the attention information
        return x


class BottleNeck(nn.Layer):
    """MobileNet倒残差瓶颈块-残差低空间+SE Attention
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 t=1,
                 n=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish,
                 use_se=False,
                 se_reduce=4):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.se_reduce = se_reduce

        self.bottle_stem = nn.Sequential(
            PointWise_Conv(in_channels=in_channels,
                           out_channels=int(t*in_channels)),
            nn.BatchNorm2D(int(t*in_channels)),
            act(),
            Linear_Depth_Separ_Conv(in_channels=int(t*in_channels),
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    act=act),
            SEBlock(in_channels=out_channels, reduce=se_reduce) if use_se else Identify()
        )

        self.bottle_blocks = []
        for i in range(0, n-1):
            self.bottle_blocks.append(
                nn.Sequential(
                    PointWise_Conv(in_channels=out_channels,
                           out_channels=int(t*out_channels)),
                    nn.BatchNorm2D(int(t*out_channels)),
                    act(),
                    Linear_Depth_Separ_Conv(in_channels=int(t*out_channels),
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=kernel_size//2,
                                            act=act),
                    SEBlock(in_channels=out_channels, reduce=se_reduce) if use_se else Identify()
                )
            )
        self.bottle_blocks = nn.LayerList(self.bottle_blocks)

    def forward(self, inputs):
        x = self.bottle_stem(inputs)
        
        for b in self.bottle_blocks:
            x = b(x) + x
        
        return x


class Stem(nn.Layer):
    """渐入层
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.act(self.bn(x))
        return x


class Classifier_Head(nn.Layer):
    """分类头
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes):
        super(Classifier_Head, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1 = PointWise_Conv(in_channels=in_channels,
                                    out_channels=hidden_channels)
        self.conv2 = PointWise_Conv(in_channels=hidden_channels,
                                    out_channels=num_classes)
        self.flatten = nn.Flatten()

        self.act1 = nn.Hardswish()
        self.act2 = nn.Softmax()
    
    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.act2(x)
        return x


class MobileNetV3_Large(nn.Layer):
    """MobileNetV3_Large实现
        Params Info:
            num_classes: 分类数
            in_channels: 输入图像通道数
            alpha: 模型伸缩大小(0.0, 1.0), 建议值:1.0, 0.75, 0.5, 0.25
            reduce: SE Attention伸缩比例
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 alpha=1.0,
                 reduce=4):
        super(MobileNetV3_Large, self).__init__()

        self.net_channels = [
                                # Config Params:
                                # in_channels, out_channels, kernel_size
                                # stride, padding,
                                # t, n,
                                # act_type, use_se
                                [in_channels, int(alpha*16), 3, 2, 1, 0, 1, nn.Hardswish, False], # Stem Conifg -- t = 0, n = 1
                                [int(alpha*16), int(alpha*16), 3, 1, 1, 1, 1, nn.ReLU, False], # first bottleneck -- t = 1, n = 1
                                [int(alpha*16), int(alpha*24), 3, 2, 1, 4, 1, nn.ReLU, False], # second bottleneck -- t = 4, n = 1
                                [int(alpha*24), int(alpha*24), 3, 1, 1, 3, 1, nn.ReLU, False], # third bottleneck -- t = 3, n = 1
                                [int(alpha*24), int(alpha*40), 5, 2, 2, 3, 3, nn.ReLU, True], # forth bottleneck -- t = 3, n = 3
                                [int(alpha*40), int(alpha*80), 3, 2, 1, 6, 1, nn.ReLU, False], # fifth bottleneck -- t = 6, n = 1
                                [int(alpha*80), int(alpha*80), 3, 1, 1, 200/80, 1, nn.Hardswish, False], # sixth bottleneck -- t = 200/80, n = 1
                                [int(alpha*80), int(alpha*80), 3, 1, 1, 184/80, 2, nn.Hardswish, False], # seventh bottleneck -- t = 184/80, n = 2
                                [int(alpha*80), int(alpha*112), 3, 1, 1, 6, 2, nn.Hardswish, True], # eighth bottleneck -- t = 6, n = 2
                                [int(alpha*112), int(alpha*160), 5, 2, 2, 6, 3, nn.Hardswish, True], # ninth bottleneck -- t = 6, n = 3
                                [int(alpha*160), int(alpha*960), 1, 1, 0, 0, 1, nn.Hardswish, False], # end conv -- t = 0, n = 1
                            ]

        self.stem = Stem(in_channels=self.net_channels[0][0],
                         out_channels=self.net_channels[0][1],
                         kernel_size=self.net_channels[0][2],
                         stride=self.net_channels[0][3],
                         padding=self.net_channels[0][4],
                         act=self.net_channels[0][7])
        
        self.mb_blocks = []
        for i in range(1, len(self.net_channels) - 1):
            self.mb_blocks.append(
                BottleNeck(
                    in_channels=self.net_channels[i][0],
                    out_channels=self.net_channels[i][1],
                    t=self.net_channels[i][5],
                    n=self.net_channels[i][6],
                    kernel_size=self.net_channels[i][2],
                    stride=self.net_channels[i][3],
                    padding=self.net_channels[i][4],
                    act=self.net_channels[i][7],
                    use_se=self.net_channels[i][8],
                    se_reduce=reduce
                )
            )
        self.mb_blocks = nn.LayerList(self.mb_blocks)

        self.end_conv = nn.Sequential(
            PointWise_Conv(in_channels=self.net_channels[-1][0],
                           out_channels=self.net_channels[-1][1]),
            nn.BatchNorm2D(self.net_channels[-1][1]),
            nn.Hardswish()
        )

        self.head = Classifier_Head(in_channels=self.net_channels[-1][1],
                                    hidden_channels=1280,
                                    num_classes=num_classes)

    def forward(self, inputs):
        x = self.stem(inputs)

        for b in self.mb_blocks:
            x = b(x)
        
        x = self.end_conv(x)
        x = self.head(x)

        return x


class MobileNetV3_Small(nn.Layer):
    """MobileNetV3_Small实现
        Params Info:
            num_classes: 分类数
            in_channels: 输入图像通道数
            alpha: 模型伸缩大小(0.0, 1.0), 建议值:1.0, 0.75, 0.5, 0.25
            reduce: SE Attention伸缩比例
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 alpha=1.0,
                 reduce=4):
        super(MobileNetV3_Small, self).__init__()

        self.net_channels = [
                                # Config Params:
                                # in_channels, out_channels, kernel_size
                                # stride, padding,
                                # t, n,
                                # act_type, use_se
                                [in_channels, int(alpha*16), 3, 2, 1, 0, 1, nn.Hardswish, False], # Stem Conifg -- t = 0, n = 1
                                [int(alpha*16), int(alpha*16), 3, 2, 1, 1, 1, nn.ReLU, True], # first bottleneck -- t = 1, n = 1
                                [int(alpha*16), int(alpha*24), 3, 2, 1, 72/16, 1, nn.ReLU, False], # second bottleneck -- t = 4.5, n = 1
                                [int(alpha*24), int(alpha*24), 3, 1, 1, 88/24, 1, nn.ReLU, False], # third bottleneck -- t = 88/24, n = 1
                                [int(alpha*24), int(alpha*40), 5, 2, 2, 4, 1, nn.Hardswish, True], # forth bottleneck -- t = 4, n = 1
                                [int(alpha*40), int(alpha*40), 5, 1, 2, 6, 2, nn.Hardswish, True], # fifth bottleneck -- t = 6, n = 2
                                [int(alpha*40), int(alpha*48), 5, 1, 2, 3, 2, nn.Hardswish, True], # sixth bottleneck -- t = 3, n = 2
                                [int(alpha*48), int(alpha*96), 5, 2, 2, 6, 3, nn.Hardswish, True], # seventh bottleneck -- t = 6, n = 3
                                [int(alpha*96), int(alpha*576), 1, 1, 0, 0, 1, nn.Hardswish, True], # end conv -- t = 0, n = 1
                            ]

        self.stem = Stem(in_channels=self.net_channels[0][0],
                         out_channels=self.net_channels[0][1],
                         kernel_size=self.net_channels[0][2],
                         stride=self.net_channels[0][3],
                         padding=self.net_channels[0][4],
                         act=self.net_channels[0][7])
        
        self.mb_blocks = []
        for i in range(1, len(self.net_channels) - 1):
            self.mb_blocks.append(
                BottleNeck(
                    in_channels=self.net_channels[i][0],
                    out_channels=self.net_channels[i][1],
                    t=self.net_channels[i][5],
                    n=self.net_channels[i][6],
                    kernel_size=self.net_channels[i][2],
                    stride=self.net_channels[i][3],
                    padding=self.net_channels[i][4],
                    act=self.net_channels[i][7],
                    use_se=self.net_channels[i][8],
                    se_reduce=reduce
                )
            )
        self.mb_blocks = nn.LayerList(self.mb_blocks)

        self.end_conv = nn.Sequential(
            PointWise_Conv(in_channels=self.net_channels[-1][0],
                           out_channels=self.net_channels[-1][1]),
            nn.BatchNorm2D(self.net_channels[-1][1]),
            nn.Hardswish(),
            SEBlock(in_channels=self.net_channels[-1][1], reduce=reduce)  # small add one se to end conv
        )

        self.head = Classifier_Head(in_channels=self.net_channels[-1][1],
                                    hidden_channels=1024,
                                    num_classes=num_classes)

    def forward(self, inputs):
        x = self.stem(inputs)

        for b in self.mb_blocks:
            x = b(x)
        
        x = self.end_conv(x)
        x = self.head(x)

        return x


def MobileNetV3_Large_for_224(num_classes,
                              in_channels):
    """构建最适合224大小图像的(大型)MobileNetV3 -- 延用V2的最适alpha
    """
    return MobileNetV3_Large(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=1.4)

def MobileNetV3_Large_Base(num_classes,
                           in_channels):
    """构建基础(大型)MobileNetV3
    """
    return MobileNetV3_Large(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=1.0)

def MobileNetV3_Large_0_75(num_classes,
                           in_channels):
    """构建较大(大型)MobileNetV2
    """
    return MobileNetV3_Large(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.75)

def MobileNetV3_Large_Mid(num_classes,
                          in_channels):
    """构建中等(大型)MobileNetV2
    """
    return MobileNetV3_Large(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.5)

def MobileNetV3_Large_Small(num_classes,
                            in_channels):
    """构建最小(大型)MobileNetV2
    """
    return MobileNetV3_Large(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.25)

def MobileNetV3_Small_for_224(num_classes,
                              in_channels):
    """构建最适合224大小图像的(小型)MobileNetV3 -- 延用V2的最适alpha
    """
    return MobileNetV3_Small(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=1.4)

def MobileNetV3_Small_Base(num_classes,
                           in_channels):
    """构建基础(小型)MobileNetV3
    """
    return MobileNetV3_Small(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=1.0)

def MobileNetV3_Small_0_75(num_classes,
                           in_channels):
    """构建较大(小型)MobileNetV2
    """
    return MobileNetV3_Small(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.75)

def MobileNetV3_Small_Mid(num_classes,
                          in_channels):
    """构建中等(小型)MobileNetV2
    """
    return MobileNetV3_Small(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.5)

def MobileNetV3_Small_Small(num_classes,
                            in_channels):
    """构建最小(小型)MobileNetV2
    """
    return MobileNetV3_Small(num_classes=num_classes,
                             in_channels=in_channels,
                             alpha=0.25)

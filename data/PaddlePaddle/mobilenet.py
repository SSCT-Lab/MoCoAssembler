import paddle
from paddle import nn
from paddle.nn import functional as F


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
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)
        
        x = self.conv(inputs)
        return x


class PointWise_Conv(nn.Layer):
    """逐点卷积-1x1
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=0):
        super(PointWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = 1
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=stride,
                              padding=padding)

    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then pointwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)

        x = self.conv(inputs)
        return x


class Depth_Separ_Conv(nn.Layer):
    """深度可分离卷积: DepthWise Conv + PointWise Conv
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU):
        super(Depth_Separ_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.depthwise_conv = DepthWise_Conv(in_channels,
                                             kernel_size,
                                             stride,
                                             padding)
        self.pointwise_conv = PointWise_Conv(in_channels,
                                             out_channels)

        self.act = act()
        self.depthwise_bn = nn.BatchNorm2D(in_channels)
        self.pointwise_bn = nn.BatchNorm2D(out_channels)

    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then pointwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)
        
        x = self.depthwise_conv(inputs)
        x = self.depthwise_bn(x)
        x = self.act(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.act(x)

        return x


class Stem(nn.Layer):
    """渐入层
    """
    def __init__(self,
                 in_channels,
                 out_channles,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 act=nn.ReLU):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channles
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channles,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2D(out_channles)
        self.act = act()
        
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then pointwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


class Classifier_Head(nn.Layer):
    """分类头
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        super(Classifier_Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=num_classes)
        self.act = nn.Softmax()
        self.flatten = nn.Flatten()
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then pointwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)

        x = self.avg_pool(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act(x)
        return x

class MobileNet(nn.Layer):
    """MobileNet实现
        Params Info:
            num_classes: 分类数
            in_channels: 输入图像通道数
            alpha: 模型伸缩大小(0.0, 1.0), 建议值:1.0, 0.75, 0.5, 0.25
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 alpha=1.0):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.alpha = alpha
        self.net_channels = [
            # item block config include:
            # in_channels, out_channels, stride, padding
            [3, int(32*alpha), 2, 1],      # Stem config
            [int(32*alpha), int(64*alpha), 1, 1],     # first conv block
            [int(64*alpha), int(128*alpha), 2, 1],    # second conv block
            [int(128*alpha), int(128*alpha), 1, 1],   # third conv block
            [int(128*alpha), int(256*alpha), 2, 1],   # forth conv block
            [int(256*alpha), int(256*alpha), 1, 1],   # fifth conv block
            [int(256*alpha), int(512*alpha), 2, 1],   # sixth conv block
            [int(512*alpha), int(512*alpha), 1, 1],   # seventh conv block
            [int(512*alpha), int(512*alpha), 1, 1],
            [int(512*alpha), int(512*alpha), 1, 1],
            [int(512*alpha), int(512*alpha), 1, 1],
            [int(512*alpha), int(512*alpha), 1, 1],
            [int(512*alpha), int(1024*alpha), 2, 1],  # eighth conv block
            [int(1024*alpha), int(1024*alpha), 1, 1]  # ninth conv block
        ]

        self.stem = Stem(in_channels=self.net_channels[0][0],
                         out_channles=self.net_channels[0][1],
                         kernel_size=3,
                         stride=self.net_channels[0][2],
                         padding=self.net_channels[0][3])
        
        self.mb_blocks = []
        for i in range(1, len(self.net_channels)):
            self.mb_blocks.append(
                Depth_Separ_Conv(in_channels=self.net_channels[i][0],
                                 out_channels=self.net_channels[i][1],
                                 kernel_size=3,
                                 stride=self.net_channels[i][2],
                                 padding=self.net_channels[i][3])
            )
        self.mb_blocks = nn.LayerList(self.mb_blocks)

        self.head = Classifier_Head(in_channels=self.net_channels[-1][1],
                                    num_classes=num_classes)

    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then pointwise inputs'shape[0](now:{0}) is equal to the conv in_channels(now:{1}).".format(inputs.shape[0], self.in_channels)
        
        x = self.stem(inputs)

        for i in range(1, len(self.net_channels)):
            x = self.mb_blocks[i-1](x)
        
        x = self.head(x)

        return x


def MobileNet_Base(num_classes,
                   in_channels):
    """构建基础(大型)MobileNet
    """
    return MobileNet(num_classes=num_classes,
                     in_channels=in_channels,
                     alpha=1.0)

def MobileNet_0_75(num_classes,
                   in_channels):
    """构建较大MobileNet
    """
    return MobileNet(num_classes=num_classes,
                     in_channels=in_channels,
                     alpha=0.75)

def MobileNet_Mid(num_classes,
                   in_channels):
    """构建中等MobileNet
    """
    return MobileNet(num_classes=num_classes,
                     in_channels=in_channels,
                     alpha=0.5)

def MobileNet_Small(num_classes,
                   in_channels):
    """构建最小MobileNet
    """
    return MobileNet(num_classes=num_classes,
                     in_channels=in_channels,
                     alpha=0.25)

import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import math
class ConvLayer0(paddle.nn.Layer):
    def __init__(self, bn0_num_channels):
        super(ConvLayer0, self).__init__()
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.tanh0(x1)
        return x2

class ConvLayer1(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels):
        super(ConvLayer1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
    def forward(self, x0):
        x1 = self.conv0(x0)
        return x1

class ConvLayer2(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class DWConvLayer(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_groups, conv0_in_channels, bn0_num_channels):
        super(DWConvLayer, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=conv0_groups, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class CombConvLayer0(paddle.nn.Layer):
    def __init__(self, convlayer10_conv0_out_channels, convlayer10_conv0_in_channels, convlayer00_bn0_num_channels, dwconvlayer0_conv0_out_channels, dwconvlayer0_conv0_groups, dwconvlayer0_conv0_in_channels, dwconvlayer0_bn0_num_channels):
        super(CombConvLayer0, self).__init__()
        self.convlayer10 = ConvLayer1(conv0_out_channels=convlayer10_conv0_out_channels, conv0_in_channels=convlayer10_conv0_in_channels)
        self.convlayer00 = ConvLayer0(bn0_num_channels=convlayer00_bn0_num_channels)
        self.dwconvlayer0 = DWConvLayer(conv0_out_channels=dwconvlayer0_conv0_out_channels, conv0_groups=dwconvlayer0_conv0_groups, conv0_in_channels=dwconvlayer0_conv0_in_channels, bn0_num_channels=dwconvlayer0_bn0_num_channels)
    def forward(self, x0):
        x1 = self.convlayer10(x0)
        x2 = self.convlayer00(x1)
        x3 = self.dwconvlayer0(x2)
        return x3

class CombConvLayer1(paddle.nn.Layer):
    def __init__(self, convlayer20_conv0_out_channels, convlayer20_conv0_in_channels, convlayer20_bn0_num_channels, dwconvlayer0_conv0_out_channels, dwconvlayer0_conv0_groups, dwconvlayer0_conv0_in_channels, dwconvlayer0_bn0_num_channels):
        super(CombConvLayer1, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=convlayer20_conv0_out_channels, conv0_in_channels=convlayer20_conv0_in_channels, bn0_num_channels=convlayer20_bn0_num_channels)
        self.dwconvlayer0 = DWConvLayer(conv0_out_channels=dwconvlayer0_conv0_out_channels, conv0_groups=dwconvlayer0_conv0_groups, conv0_in_channels=dwconvlayer0_conv0_in_channels, bn0_num_channels=dwconvlayer0_bn0_num_channels)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.dwconvlayer0(x1)
        return x2

class ModuleList3(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList3, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=14, convlayer20_conv0_in_channels=64, convlayer20_bn0_num_channels=14, dwconvlayer0_conv0_out_channels=14, dwconvlayer0_conv0_groups=14, dwconvlayer0_conv0_in_channels=14, dwconvlayer0_bn0_num_channels=14)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList4(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList4, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=24, convlayer10_conv0_in_channels=78, convlayer00_bn0_num_channels=24, dwconvlayer0_conv0_out_channels=24, dwconvlayer0_conv0_groups=24, dwconvlayer0_conv0_in_channels=24, dwconvlayer0_bn0_num_channels=24)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=14, convlayer20_conv0_in_channels=24, convlayer20_bn0_num_channels=14, dwconvlayer0_conv0_out_channels=14, dwconvlayer0_conv0_groups=14, dwconvlayer0_conv0_in_channels=14, dwconvlayer0_bn0_num_channels=14)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList5(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList5, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=40, convlayer10_conv0_in_channels=102, convlayer00_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=14, convlayer20_conv0_in_channels=40, convlayer20_bn0_num_channels=14, dwconvlayer0_conv0_out_channels=14, dwconvlayer0_conv0_groups=14, dwconvlayer0_conv0_in_channels=14, dwconvlayer0_bn0_num_channels=14)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList6(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList6, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=24, convlayer10_conv0_in_channels=54, convlayer00_bn0_num_channels=24, dwconvlayer0_conv0_out_channels=24, dwconvlayer0_conv0_groups=24, dwconvlayer0_conv0_in_channels=24, dwconvlayer0_bn0_num_channels=24)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=14, convlayer20_conv0_in_channels=24, convlayer20_bn0_num_channels=14, dwconvlayer0_conv0_out_channels=14, dwconvlayer0_conv0_groups=14, dwconvlayer0_conv0_in_channels=14, dwconvlayer0_bn0_num_channels=14)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList7(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList7, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=68, convlayer10_conv0_in_channels=142, convlayer00_bn0_num_channels=68, dwconvlayer0_conv0_out_channels=68, dwconvlayer0_conv0_groups=68, dwconvlayer0_conv0_in_channels=68, dwconvlayer0_bn0_num_channels=68)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList8(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList8, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=128, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList9(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList9, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=28, convlayer10_conv0_in_channels=144, convlayer00_bn0_num_channels=28, dwconvlayer0_conv0_out_channels=28, dwconvlayer0_conv0_groups=28, dwconvlayer0_conv0_in_channels=28, dwconvlayer0_bn0_num_channels=28)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=28, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList10(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList10, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=46, convlayer10_conv0_in_channels=172, convlayer00_bn0_num_channels=46, dwconvlayer0_conv0_out_channels=46, dwconvlayer0_conv0_groups=46, dwconvlayer0_conv0_in_channels=46, dwconvlayer0_bn0_num_channels=46)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=46, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList11(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList11, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=28, convlayer10_conv0_in_channels=62, convlayer00_bn0_num_channels=28, dwconvlayer0_conv0_out_channels=28, dwconvlayer0_conv0_groups=28, dwconvlayer0_conv0_in_channels=28, dwconvlayer0_bn0_num_channels=28)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=28, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList12(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList12, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=78, convlayer10_conv0_in_channels=218, convlayer00_bn0_num_channels=78, dwconvlayer0_conv0_out_channels=78, dwconvlayer0_conv0_groups=78, dwconvlayer0_conv0_in_channels=78, dwconvlayer0_bn0_num_channels=78)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=78, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList13(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList13, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=28, convlayer10_conv0_in_channels=94, convlayer00_bn0_num_channels=28, dwconvlayer0_conv0_out_channels=28, dwconvlayer0_conv0_groups=28, dwconvlayer0_conv0_in_channels=28, dwconvlayer0_bn0_num_channels=28)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=28, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList14(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList14, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=46, convlayer10_conv0_in_channels=122, convlayer00_bn0_num_channels=46, dwconvlayer0_conv0_out_channels=46, dwconvlayer0_conv0_groups=46, dwconvlayer0_conv0_in_channels=46, dwconvlayer0_bn0_num_channels=46)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=46, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList15(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList15, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=28, convlayer10_conv0_in_channels=62, convlayer00_bn0_num_channels=28, dwconvlayer0_conv0_out_channels=28, dwconvlayer0_conv0_groups=28, dwconvlayer0_conv0_in_channels=28, dwconvlayer0_bn0_num_channels=28)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=28, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList16(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList16, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=134, convlayer10_conv0_in_channels=296, convlayer00_bn0_num_channels=134, dwconvlayer0_conv0_out_channels=134, dwconvlayer0_conv0_groups=134, dwconvlayer0_conv0_in_channels=134, dwconvlayer0_bn0_num_channels=134)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList17(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList17, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=256, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList18(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList18, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=34, convlayer10_conv0_in_channels=276, convlayer00_bn0_num_channels=34, dwconvlayer0_conv0_out_channels=34, dwconvlayer0_conv0_groups=34, dwconvlayer0_conv0_in_channels=34, dwconvlayer0_bn0_num_channels=34)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=34, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList19(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList19, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=58, convlayer10_conv0_in_channels=310, convlayer00_bn0_num_channels=58, dwconvlayer0_conv0_out_channels=58, dwconvlayer0_conv0_groups=58, dwconvlayer0_conv0_in_channels=58, dwconvlayer0_bn0_num_channels=58)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=58, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList20(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList20, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=34, convlayer10_conv0_in_channels=78, convlayer00_bn0_num_channels=34, dwconvlayer0_conv0_out_channels=34, dwconvlayer0_conv0_groups=34, dwconvlayer0_conv0_in_channels=34, dwconvlayer0_bn0_num_channels=34)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=34, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList21(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList21, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=98, convlayer10_conv0_in_channels=368, convlayer00_bn0_num_channels=98, dwconvlayer0_conv0_out_channels=98, dwconvlayer0_conv0_groups=98, dwconvlayer0_conv0_in_channels=98, dwconvlayer0_bn0_num_channels=98)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=98, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList22(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList22, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=34, convlayer10_conv0_in_channels=118, convlayer00_bn0_num_channels=34, dwconvlayer0_conv0_out_channels=34, dwconvlayer0_conv0_groups=34, dwconvlayer0_conv0_in_channels=34, dwconvlayer0_bn0_num_channels=34)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=34, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList23(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList23, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=58, convlayer10_conv0_in_channels=152, convlayer00_bn0_num_channels=58, dwconvlayer0_conv0_out_channels=58, dwconvlayer0_conv0_groups=58, dwconvlayer0_conv0_in_channels=58, dwconvlayer0_bn0_num_channels=58)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=58, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList24(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList24, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=34, convlayer10_conv0_in_channels=78, convlayer00_bn0_num_channels=34, dwconvlayer0_conv0_out_channels=34, dwconvlayer0_conv0_groups=34, dwconvlayer0_conv0_in_channels=34, dwconvlayer0_bn0_num_channels=34)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=34, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList25(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList25, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=168, convlayer10_conv0_in_channels=466, convlayer00_bn0_num_channels=168, dwconvlayer0_conv0_out_channels=168, dwconvlayer0_conv0_groups=168, dwconvlayer0_conv0_in_channels=168, dwconvlayer0_bn0_num_channels=168)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList26(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList26, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=320, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList27(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList27, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=68, convlayer10_conv0_in_channels=360, convlayer00_bn0_num_channels=68, dwconvlayer0_conv0_out_channels=68, dwconvlayer0_conv0_groups=68, dwconvlayer0_conv0_in_channels=68, dwconvlayer0_bn0_num_channels=68)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=68, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList28(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList28, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=116, convlayer10_conv0_in_channels=428, convlayer00_bn0_num_channels=116, dwconvlayer0_conv0_out_channels=116, dwconvlayer0_conv0_groups=116, dwconvlayer0_conv0_in_channels=116, dwconvlayer0_bn0_num_channels=116)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=116, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList29(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList29, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=68, convlayer10_conv0_in_channels=156, convlayer00_bn0_num_channels=68, dwconvlayer0_conv0_out_channels=68, dwconvlayer0_conv0_groups=68, dwconvlayer0_conv0_in_channels=68, dwconvlayer0_bn0_num_channels=68)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=68, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList30(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList30, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=196, convlayer10_conv0_in_channels=544, convlayer00_bn0_num_channels=196, dwconvlayer0_conv0_out_channels=196, dwconvlayer0_conv0_groups=196, dwconvlayer0_conv0_in_channels=196, dwconvlayer0_bn0_num_channels=196)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=196, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList31(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList31, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=68, convlayer10_conv0_in_channels=236, convlayer00_bn0_num_channels=68, dwconvlayer0_conv0_out_channels=68, dwconvlayer0_conv0_groups=68, dwconvlayer0_conv0_in_channels=68, dwconvlayer0_bn0_num_channels=68)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=68, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList32(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList32, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=116, convlayer10_conv0_in_channels=304, convlayer00_bn0_num_channels=116, dwconvlayer0_conv0_out_channels=116, dwconvlayer0_conv0_groups=116, dwconvlayer0_conv0_in_channels=116, dwconvlayer0_bn0_num_channels=116)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=116, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList33(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList33, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=68, convlayer10_conv0_in_channels=156, convlayer00_bn0_num_channels=68, dwconvlayer0_conv0_out_channels=68, dwconvlayer0_conv0_groups=68, dwconvlayer0_conv0_in_channels=68, dwconvlayer0_bn0_num_channels=68)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=40, convlayer20_conv0_in_channels=68, convlayer20_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList34(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList34, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=334, convlayer10_conv0_in_channels=740, convlayer00_bn0_num_channels=334, dwconvlayer0_conv0_out_channels=334, dwconvlayer0_conv0_groups=334, dwconvlayer0_conv0_in_channels=334, dwconvlayer0_bn0_num_channels=334)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList35(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList35, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=160, convlayer20_conv0_in_channels=640, convlayer20_bn0_num_channels=160, dwconvlayer0_conv0_out_channels=160, dwconvlayer0_conv0_groups=160, dwconvlayer0_conv0_in_channels=160, dwconvlayer0_bn0_num_channels=160)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList36(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList36, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=272, convlayer10_conv0_in_channels=800, convlayer00_bn0_num_channels=272, dwconvlayer0_conv0_out_channels=272, dwconvlayer0_conv0_groups=272, dwconvlayer0_conv0_in_channels=272, dwconvlayer0_bn0_num_channels=272)
        self.combconvlayer10 = CombConvLayer1(convlayer20_conv0_out_channels=160, convlayer20_conv0_in_channels=272, convlayer20_bn0_num_channels=160, dwconvlayer0_conv0_out_channels=160, dwconvlayer0_conv0_groups=160, dwconvlayer0_conv0_in_channels=160, dwconvlayer0_bn0_num_channels=160)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        x2 = self.combconvlayer10(x1)
        return x1, x2

class ModuleList37(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList37, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer10_conv0_out_channels=462, convlayer10_conv0_in_channels=1072, convlayer00_bn0_num_channels=462, dwconvlayer0_conv0_out_channels=462, dwconvlayer0_conv0_groups=462, dwconvlayer0_conv0_in_channels=462, dwconvlayer0_bn0_num_channels=462)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ConvLayer0_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=640, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=654)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=640, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=328)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=320, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer2_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer3(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=262)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=32)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=124)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer6(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer6, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=782)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class DWConvLayer0(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=640, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=640, in_channels=640)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=640, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer1(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, in_channels=128)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer2(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=320, in_channels=320)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=320, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer3(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=64, in_channels=64)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class HarDBlock0(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock0, self).__init__()
        self.modulelist30 = ModuleList3()
        self.modulelist40 = ModuleList4()
        self.modulelist50 = ModuleList5()
        self.modulelist60 = ModuleList6()
        self.modulelist70 = ModuleList7()
    def forward(self, x0):
        x1 = self.modulelist30(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist40(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist50(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist60(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16 = self.modulelist70(x15)
        x17 = [x1, x5, x9, x13, x16]
        x18 = paddle.concat(x=x17, axis=1)
        return x18

class HarDBlock1(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock1, self).__init__()
        self.modulelist80 = ModuleList8()
        self.modulelist90 = ModuleList9()
        self.modulelist100 = ModuleList10()
        self.modulelist110 = ModuleList11()
        self.modulelist120 = ModuleList12()
        self.modulelist130 = ModuleList13()
        self.modulelist140 = ModuleList14()
        self.modulelist150 = ModuleList15()
        self.modulelist160 = ModuleList16()
    def forward(self, x0):
        x1 = self.modulelist80(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist90(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist100(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist110(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist120(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist130(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist140(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist150(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist160(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock2(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock2, self).__init__()
        self.modulelist260 = ModuleList26()
        self.modulelist270 = ModuleList27()
        self.modulelist280 = ModuleList28()
        self.modulelist290 = ModuleList29()
        self.modulelist300 = ModuleList30()
        self.modulelist310 = ModuleList31()
        self.modulelist320 = ModuleList32()
        self.modulelist330 = ModuleList33()
        self.modulelist340 = ModuleList34()
    def forward(self, x0):
        x1 = self.modulelist260(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist270(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist280(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist290(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist300(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist310(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist320(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist330(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist340(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock3(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock3, self).__init__()
        self.modulelist350 = ModuleList35()
        self.modulelist360 = ModuleList36()
        self.modulelist370 = ModuleList37()
    def forward(self, x0):
        x1 = self.modulelist350(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist360(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8 = self.modulelist370(x7)
        x9 = [x1, x5, x8]
        x10 = paddle.concat(x=x9, axis=1)
        return x10

class HarDBlock4(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock4, self).__init__()
        self.modulelist170 = ModuleList17()
        self.modulelist180 = ModuleList18()
        self.modulelist190 = ModuleList19()
        self.modulelist200 = ModuleList20()
        self.modulelist210 = ModuleList21()
        self.modulelist220 = ModuleList22()
        self.modulelist230 = ModuleList23()
        self.modulelist240 = ModuleList24()
        self.modulelist250 = ModuleList25()
    def forward(self, x0):
        x1 = self.modulelist170(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist180(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist190(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist200(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist210(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist220(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist230(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist240(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist250(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class Base_16(paddle.nn.Layer):
    def __init__(self, ):
        super(Base_16, self).__init__()
        self.x1 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear0 = paddle.nn.Linear(in_features=1024, out_features=1000)
    def forward(self, x0):
        x2 = self.x1(x0)
        x3 = paddle.reshape(x=x2, shape=[x2.shape[0], -1])#resahpe 
        x4 = self.dropout0(x3)
        x5 = self.linear0(x4)
        return x5

class ModuleList(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList, self).__init__()
        self.convlayer2_00 = ConvLayer2_0()
        self.convlayer40 = ConvLayer4()
        self.dwconvlayer30 = DWConvLayer3()
        self.hardblock00 = HarDBlock0()
        self.convlayer50 = ConvLayer5()
        self.dwconvlayer10 = DWConvLayer1()
        self.hardblock10 = HarDBlock1()
        self.convlayer30 = ConvLayer3()
        self.hardblock40 = HarDBlock4()
        self.convlayer1_00 = ConvLayer1_0()
        self.dwconvlayer20 = DWConvLayer2()
        self.hardblock20 = HarDBlock2()
        self.convlayer0_00 = ConvLayer0_0()
        self.dwconvlayer00 = DWConvLayer0()
        self.hardblock30 = HarDBlock3()
        self.convlayer60 = ConvLayer6()
        self.base_160 = Base_16()
    def forward(self, x0):
        x1 = self.convlayer2_00(x0)
        x2 = self.convlayer40(x1)
        x3 = self.dwconvlayer30(x2)
        x4 = self.hardblock00(x3)
        x5 = self.convlayer50(x4)
        x6 = self.dwconvlayer10(x5)
        x7 = self.hardblock10(x6)
        x8 = self.convlayer30(x7)
        x9 = self.hardblock40(x8)
        x10 = self.convlayer1_00(x9)
        x11 = self.dwconvlayer20(x10)
        x12 = self.hardblock20(x11)
        x13 = self.convlayer0_00(x12)
        x14 = self.dwconvlayer00(x13)
        x15 = self.hardblock30(x14)
        x16 = self.convlayer60(x15)
        x17 = self.base_160(x16)
        return x17

class HarDNet68ds(paddle.nn.Layer):
    def __init__(self, class_dim=1000):
        super(HarDNet68ds, self).__init__()
        self.modulelist0 = ModuleList()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.modulelist0(x0)
        return x1

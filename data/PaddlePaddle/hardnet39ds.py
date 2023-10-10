import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import math
class ConvLayer0(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels):
        super(ConvLayer0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
    def forward(self, x0):
        x1 = self.conv0(x0)
        return x1

class ConvLayer1(paddle.nn.Layer):
    def __init__(self, bn0_num_channels):
        super(ConvLayer1, self).__init__()
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.bn0(x0)
        x2 = self.tanh0(x1)
        return x2

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
    def __init__(self, convlayer20_conv0_out_channels, convlayer20_conv0_in_channels, convlayer20_bn0_num_channels, dwconvlayer0_conv0_out_channels, dwconvlayer0_conv0_groups, dwconvlayer0_conv0_in_channels, dwconvlayer0_bn0_num_channels):
        super(CombConvLayer0, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=convlayer20_conv0_out_channels, conv0_in_channels=convlayer20_conv0_in_channels, bn0_num_channels=convlayer20_bn0_num_channels)
        self.dwconvlayer0 = DWConvLayer(conv0_out_channels=dwconvlayer0_conv0_out_channels, conv0_groups=dwconvlayer0_conv0_groups, conv0_in_channels=dwconvlayer0_conv0_in_channels, bn0_num_channels=dwconvlayer0_bn0_num_channels)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.dwconvlayer0(x1)
        return x2

class CombConvLayer1(paddle.nn.Layer):
    def __init__(self, convlayer00_conv0_out_channels, convlayer00_conv0_in_channels, convlayer10_bn0_num_channels, dwconvlayer0_conv0_out_channels, dwconvlayer0_conv0_groups, dwconvlayer0_conv0_in_channels, dwconvlayer0_bn0_num_channels):
        super(CombConvLayer1, self).__init__()
        self.convlayer00 = ConvLayer0(conv0_out_channels=convlayer00_conv0_out_channels, conv0_in_channels=convlayer00_conv0_in_channels)
        self.convlayer10 = ConvLayer1(bn0_num_channels=convlayer10_bn0_num_channels)
        self.dwconvlayer0 = DWConvLayer(conv0_out_channels=dwconvlayer0_conv0_out_channels, conv0_groups=dwconvlayer0_conv0_groups, conv0_in_channels=dwconvlayer0_conv0_in_channels, bn0_num_channels=dwconvlayer0_bn0_num_channels)
    def forward(self, x0):
        x1 = self.convlayer00(x0)
        x2 = self.convlayer10(x1)
        x3 = self.dwconvlayer0(x2)
        return x3

class ModuleList3(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList3, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=48, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList4(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList4, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=26, convlayer00_conv0_in_channels=64, convlayer10_bn0_num_channels=26, dwconvlayer0_conv0_out_channels=26, dwconvlayer0_conv0_groups=26, dwconvlayer0_conv0_in_channels=26, dwconvlayer0_bn0_num_channels=26)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=16, convlayer20_conv0_in_channels=26, convlayer20_bn0_num_channels=16, dwconvlayer0_conv0_out_channels=16, dwconvlayer0_conv0_groups=16, dwconvlayer0_conv0_in_channels=16, dwconvlayer0_bn0_num_channels=16)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList5(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList5, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=40, convlayer00_conv0_in_channels=90, convlayer10_bn0_num_channels=40, dwconvlayer0_conv0_out_channels=40, dwconvlayer0_conv0_groups=40, dwconvlayer0_conv0_in_channels=40, dwconvlayer0_bn0_num_channels=40)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList6(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList6, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=96, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList7(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList7, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=32, convlayer00_conv0_in_channels=116, convlayer10_bn0_num_channels=32, dwconvlayer0_conv0_out_channels=32, dwconvlayer0_conv0_groups=32, dwconvlayer0_conv0_in_channels=32, dwconvlayer0_bn0_num_channels=32)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=32, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList8(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList8, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=52, convlayer00_conv0_in_channels=148, convlayer10_bn0_num_channels=52, dwconvlayer0_conv0_out_channels=52, dwconvlayer0_conv0_groups=52, dwconvlayer0_conv0_in_channels=52, dwconvlayer0_bn0_num_channels=52)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=52, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList9(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList9, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=32, convlayer00_conv0_in_channels=72, convlayer10_bn0_num_channels=32, dwconvlayer0_conv0_out_channels=32, dwconvlayer0_conv0_groups=32, dwconvlayer0_conv0_in_channels=32, dwconvlayer0_bn0_num_channels=32)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=32, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList10(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList10, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=82, convlayer00_conv0_in_channels=200, convlayer10_bn0_num_channels=82, dwconvlayer0_conv0_out_channels=82, dwconvlayer0_conv0_groups=82, dwconvlayer0_conv0_in_channels=82, dwconvlayer0_bn0_num_channels=82)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=82, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList11(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList11, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=32, convlayer00_conv0_in_channels=102, convlayer10_bn0_num_channels=32, dwconvlayer0_conv0_out_channels=32, dwconvlayer0_conv0_groups=32, dwconvlayer0_conv0_in_channels=32, dwconvlayer0_bn0_num_channels=32)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=32, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList12(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList12, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=52, convlayer00_conv0_in_channels=134, convlayer10_bn0_num_channels=52, dwconvlayer0_conv0_out_channels=52, dwconvlayer0_conv0_groups=52, dwconvlayer0_conv0_in_channels=52, dwconvlayer0_bn0_num_channels=52)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=52, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList13(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList13, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=32, convlayer00_conv0_in_channels=72, convlayer10_bn0_num_channels=32, dwconvlayer0_conv0_out_channels=32, dwconvlayer0_conv0_groups=32, dwconvlayer0_conv0_in_channels=32, dwconvlayer0_bn0_num_channels=32)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=20, convlayer20_conv0_in_channels=32, convlayer20_bn0_num_channels=20, dwconvlayer0_conv0_out_channels=20, dwconvlayer0_conv0_groups=20, dwconvlayer0_conv0_in_channels=20, dwconvlayer0_bn0_num_channels=20)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList14(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList14, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=132, convlayer00_conv0_in_channels=282, convlayer10_bn0_num_channels=132, dwconvlayer0_conv0_out_channels=132, dwconvlayer0_conv0_groups=132, dwconvlayer0_conv0_in_channels=132, dwconvlayer0_bn0_num_channels=132)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList15(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList15, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=64, convlayer20_conv0_in_channels=320, convlayer20_bn0_num_channels=64, dwconvlayer0_conv0_out_channels=64, dwconvlayer0_conv0_groups=64, dwconvlayer0_conv0_in_channels=64, dwconvlayer0_bn0_num_channels=64)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList16(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList16, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=102, convlayer00_conv0_in_channels=384, convlayer10_bn0_num_channels=102, dwconvlayer0_conv0_out_channels=102, dwconvlayer0_conv0_groups=102, dwconvlayer0_conv0_in_channels=102, dwconvlayer0_bn0_num_channels=102)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=64, convlayer20_conv0_in_channels=102, convlayer20_bn0_num_channels=64, dwconvlayer0_conv0_out_channels=64, dwconvlayer0_conv0_groups=64, dwconvlayer0_conv0_in_channels=64, dwconvlayer0_bn0_num_channels=64)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList17(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList17, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=164, convlayer00_conv0_in_channels=486, convlayer10_bn0_num_channels=164, dwconvlayer0_conv0_out_channels=164, dwconvlayer0_conv0_groups=164, dwconvlayer0_conv0_in_channels=164, dwconvlayer0_bn0_num_channels=164)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=64, convlayer20_conv0_in_channels=164, convlayer20_bn0_num_channels=64, dwconvlayer0_conv0_out_channels=64, dwconvlayer0_conv0_groups=64, dwconvlayer0_conv0_in_channels=64, dwconvlayer0_bn0_num_channels=64)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList18(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList18, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=102, convlayer00_conv0_in_channels=228, convlayer10_bn0_num_channels=102, dwconvlayer0_conv0_out_channels=102, dwconvlayer0_conv0_groups=102, dwconvlayer0_conv0_in_channels=102, dwconvlayer0_bn0_num_channels=102)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=64, convlayer20_conv0_in_channels=102, convlayer20_bn0_num_channels=64, dwconvlayer0_conv0_out_channels=64, dwconvlayer0_conv0_groups=64, dwconvlayer0_conv0_in_channels=64, dwconvlayer0_bn0_num_channels=64)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList19(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList19, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=262, convlayer00_conv0_in_channels=650, convlayer10_bn0_num_channels=262, dwconvlayer0_conv0_out_channels=262, dwconvlayer0_conv0_groups=262, dwconvlayer0_conv0_in_channels=262, dwconvlayer0_bn0_num_channels=262)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ModuleList20(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList20, self).__init__()
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=160, convlayer20_conv0_in_channels=640, convlayer20_bn0_num_channels=160, dwconvlayer0_conv0_out_channels=160, dwconvlayer0_conv0_groups=160, dwconvlayer0_conv0_in_channels=160, dwconvlayer0_bn0_num_channels=160)
    def forward(self, x0):
        x1 = self.combconvlayer00(x0)
        return x1

class ModuleList21(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList21, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=256, convlayer00_conv0_in_channels=800, convlayer10_bn0_num_channels=256, dwconvlayer0_conv0_out_channels=256, dwconvlayer0_conv0_groups=256, dwconvlayer0_conv0_in_channels=256, dwconvlayer0_bn0_num_channels=256)
        self.combconvlayer00 = CombConvLayer0(convlayer20_conv0_out_channels=160, convlayer20_conv0_in_channels=256, convlayer20_bn0_num_channels=160, dwconvlayer0_conv0_out_channels=160, dwconvlayer0_conv0_groups=160, dwconvlayer0_conv0_in_channels=160, dwconvlayer0_bn0_num_channels=160)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        x2 = self.combconvlayer00(x1)
        return x1, x2

class ModuleList22(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList22, self).__init__()
        self.combconvlayer10 = CombConvLayer1(convlayer00_conv0_out_channels=410, convlayer00_conv0_in_channels=1056, convlayer10_bn0_num_channels=410, dwconvlayer0_conv0_out_channels=410, dwconvlayer0_conv0_groups=410, dwconvlayer0_conv0_in_channels=410, dwconvlayer0_bn0_num_channels=410)
    def forward(self, x0):
        x1 = self.combconvlayer10(x0)
        return x1

class ConvLayer0_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=96, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=72)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=96, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=292)
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
        self.conv0 = paddle.nn.Conv2D(out_channels=640, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=518)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=640, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer3(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=730)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=48, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=24)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=48, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=24, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=24, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class DWConvLayer0(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=320, in_channels=320)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=320, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer1(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=640, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=640, in_channels=640)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=640, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer2(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=96, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=96, in_channels=96)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=96, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class DWConvLayer3(paddle.nn.Layer):
    def __init__(self, ):
        super(DWConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=48, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=48, in_channels=48)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=48, momentum=0.1, epsilon=1e-05)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class HarDBlock0(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock0, self).__init__()
        self.modulelist200 = ModuleList20()
        self.modulelist210 = ModuleList21()
        self.modulelist220 = ModuleList22()
    def forward(self, x0):
        x1 = self.modulelist200(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist210(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8 = self.modulelist220(x7)
        x9 = [x1, x5, x8]
        x10 = paddle.concat(x=x9, axis=1)
        return x10

class HarDBlock1(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock1, self).__init__()
        self.modulelist60 = ModuleList6()
        self.modulelist70 = ModuleList7()
        self.modulelist80 = ModuleList8()
        self.modulelist90 = ModuleList9()
        self.modulelist100 = ModuleList10()
        self.modulelist110 = ModuleList11()
        self.modulelist120 = ModuleList12()
        self.modulelist130 = ModuleList13()
        self.modulelist140 = ModuleList14()
    def forward(self, x0):
        x1 = self.modulelist60(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist70(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist80(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist90(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist100(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist110(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist120(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist130(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist140(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock2(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock2, self).__init__()
        self.modulelist30 = ModuleList3()
        self.modulelist40 = ModuleList4()
        self.modulelist50 = ModuleList5()
    def forward(self, x0):
        x1 = self.modulelist30(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist40(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8 = self.modulelist50(x7)
        x9 = [x1, x5, x8]
        x10 = paddle.concat(x=x9, axis=1)
        return x10

class HarDBlock3(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock3, self).__init__()
        self.modulelist150 = ModuleList15()
        self.modulelist160 = ModuleList16()
        self.modulelist170 = ModuleList17()
        self.modulelist180 = ModuleList18()
        self.modulelist190 = ModuleList19()
    def forward(self, x0):
        x1 = self.modulelist150(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist160(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist170(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist180(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16 = self.modulelist190(x15)
        x17 = [x1, x5, x9, x13, x16]
        x18 = paddle.concat(x=x17, axis=1)
        return x18

class Base_14(paddle.nn.Layer):
    def __init__(self, ):
        super(Base_14, self).__init__()
        self.x1 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear0 = paddle.nn.Linear(in_features=1024, out_features=1000)
    def forward(self, x0):
        x2 = self.x1(x0)
        x3 = paddle.reshape(x=x2, shape=[x2.shape[0], -1])
        x4 = self.dropout0(x3)
        x5 = self.linear0(x4)
        return x5

class ModuleList(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList, self).__init__()
        self.convlayer50 = ConvLayer5()
        self.convlayer40 = ConvLayer4()
        self.dwconvlayer30 = DWConvLayer3()
        self.hardblock20 = HarDBlock2()
        self.convlayer0_00 = ConvLayer0_0()
        self.dwconvlayer20 = DWConvLayer2()
        self.hardblock10 = HarDBlock1()
        self.convlayer1_00 = ConvLayer1_0()
        self.dwconvlayer00 = DWConvLayer0()
        self.hardblock30 = HarDBlock3()
        self.convlayer2_00 = ConvLayer2_0()
        self.dwconvlayer10 = DWConvLayer1()
        self.hardblock00 = HarDBlock0()
        self.convlayer30 = ConvLayer3()
        self.base_140 = Base_14()
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer40(x1)
        x3 = self.dwconvlayer30(x2)
        x4 = self.hardblock20(x3)
        x5 = self.convlayer0_00(x4)
        x6 = self.dwconvlayer20(x5)
        x7 = self.hardblock10(x6)
        x8 = self.convlayer1_00(x7)
        x9 = self.dwconvlayer00(x8)
        x10 = self.hardblock30(x9)
        x11 = self.convlayer2_00(x10)
        x12 = self.dwconvlayer10(x11)
        x13 = self.hardblock00(x12)
        x14 = self.convlayer30(x13)
        x15 = self.base_140(x14)
        return x15

class HarDNet39ds(paddle.nn.Layer):
    def __init__(self, class_dim=1000):
        super(HarDNet39ds, self).__init__()
        self.modulelist0 = ModuleList()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.modulelist0(x0)
        return x1

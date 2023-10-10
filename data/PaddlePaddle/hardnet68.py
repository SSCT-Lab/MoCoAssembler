import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import math

class ConvLayer0(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=16, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=16, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=14, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=14, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer2(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer3(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=40, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=40, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer6(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer6, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=20, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=20, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer7(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer7, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer8(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer8, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=160, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=160, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer9(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer9, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ModuleList15(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList15, self).__init__()
        self.convlayer10 = ConvLayer1(conv0_in_channels=64)
    def forward(self, x0):
        x1 = self.convlayer10(x0)
        return x1

class ModuleList16(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList16, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=24, conv0_in_channels=78, bn0_num_channels=24)
        self.convlayer10 = ConvLayer1(conv0_in_channels=24)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList17(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList17, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=40, conv0_in_channels=102, bn0_num_channels=40)
        self.convlayer10 = ConvLayer1(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList18(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList18, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=24, conv0_in_channels=54, bn0_num_channels=24)
        self.convlayer10 = ConvLayer1(conv0_in_channels=24)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList19(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList19, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=68, conv0_in_channels=142, bn0_num_channels=68)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        return x1

class ModuleList20(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList20, self).__init__()
        self.convlayer00 = ConvLayer0(conv0_in_channels=128)
    def forward(self, x0):
        x1 = self.convlayer00(x0)
        return x1

class ModuleList21(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList21, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=28, conv0_in_channels=144, bn0_num_channels=28)
        self.convlayer00 = ConvLayer0(conv0_in_channels=28)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList22(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList22, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=46, conv0_in_channels=172, bn0_num_channels=46)
        self.convlayer00 = ConvLayer0(conv0_in_channels=46)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList23(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList23, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=28, conv0_in_channels=62, bn0_num_channels=28)
        self.convlayer00 = ConvLayer0(conv0_in_channels=28)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList24(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList24, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=78, conv0_in_channels=218, bn0_num_channels=78)
        self.convlayer00 = ConvLayer0(conv0_in_channels=78)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList25(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList25, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=28, conv0_in_channels=94, bn0_num_channels=28)
        self.convlayer00 = ConvLayer0(conv0_in_channels=28)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList26(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList26, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=46, conv0_in_channels=122, bn0_num_channels=46)
        self.convlayer00 = ConvLayer0(conv0_in_channels=46)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList27(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList27, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=28, conv0_in_channels=62, bn0_num_channels=28)
        self.convlayer00 = ConvLayer0(conv0_in_channels=28)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList28(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList28, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_out_channels=134, conv0_in_channels=296, bn0_num_channels=134)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        return x1

class ModuleList29(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList29, self).__init__()
        self.convlayer60 = ConvLayer6(conv0_in_channels=256)
    def forward(self, x0):
        x1 = self.convlayer60(x0)
        return x1

class ModuleList30(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList30, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=34, conv0_in_channels=276, bn0_num_channels=34)
        self.convlayer60 = ConvLayer6(conv0_in_channels=34)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList31(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList31, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=58, conv0_in_channels=310, bn0_num_channels=58)
        self.convlayer60 = ConvLayer6(conv0_in_channels=58)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList32(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList32, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=34, conv0_in_channels=78, bn0_num_channels=34)
        self.convlayer60 = ConvLayer6(conv0_in_channels=34)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList33(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList33, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=98, conv0_in_channels=368, bn0_num_channels=98)
        self.convlayer60 = ConvLayer6(conv0_in_channels=98)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList34(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList34, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=34, conv0_in_channels=118, bn0_num_channels=34)
        self.convlayer60 = ConvLayer6(conv0_in_channels=34)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList35(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList35, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=58, conv0_in_channels=152, bn0_num_channels=58)
        self.convlayer60 = ConvLayer6(conv0_in_channels=58)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList36(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList36, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=34, conv0_in_channels=78, bn0_num_channels=34)
        self.convlayer60 = ConvLayer6(conv0_in_channels=34)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList37(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList37, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_out_channels=168, conv0_in_channels=466, bn0_num_channels=168)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        return x1

class ModuleList38(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList38, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_in_channels=320)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        return x1

class ModuleList39(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList39, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=68, conv0_in_channels=360, bn0_num_channels=68)
        self.convlayer30 = ConvLayer3(conv0_in_channels=68)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList40(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList40, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=116, conv0_in_channels=428, bn0_num_channels=116)
        self.convlayer30 = ConvLayer3(conv0_in_channels=116)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList41(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList41, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=68, conv0_in_channels=156, bn0_num_channels=68)
        self.convlayer30 = ConvLayer3(conv0_in_channels=68)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList42(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList42, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=196, conv0_in_channels=544, bn0_num_channels=196)
        self.convlayer30 = ConvLayer3(conv0_in_channels=196)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList43(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList43, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=68, conv0_in_channels=236, bn0_num_channels=68)
        self.convlayer30 = ConvLayer3(conv0_in_channels=68)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList44(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList44, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=116, conv0_in_channels=304, bn0_num_channels=116)
        self.convlayer30 = ConvLayer3(conv0_in_channels=116)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList45(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList45, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=68, conv0_in_channels=156, bn0_num_channels=68)
        self.convlayer30 = ConvLayer3(conv0_in_channels=68)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        x2 = self.convlayer30(x1)
        return x1, x2

class ModuleList46(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList46, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_out_channels=334, conv0_in_channels=740, bn0_num_channels=334)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        return x1

class ModuleList47(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList47, self).__init__()
        self.convlayer80 = ConvLayer8(conv0_in_channels=640)
    def forward(self, x0):
        x1 = self.convlayer80(x0)
        return x1

class ModuleList48(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList48, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=272, conv0_in_channels=800, bn0_num_channels=272)
        self.convlayer80 = ConvLayer8(conv0_in_channels=272)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer80(x1)
        return x1, x2

class ModuleList49(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList49, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=462, conv0_in_channels=1072, bn0_num_channels=462)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        return x1

class ConvLayer0_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=782)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=32)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer2_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=262)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer3_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer3_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=640, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=654)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=640, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer4_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=124)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=128, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer5_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=328)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=320, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer6_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer6_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=32, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class HarDBlock0(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock0, self).__init__()
        self.modulelist380 = ModuleList38()
        self.modulelist390 = ModuleList39()
        self.modulelist400 = ModuleList40()
        self.modulelist410 = ModuleList41()
        self.modulelist420 = ModuleList42()
        self.modulelist430 = ModuleList43()
        self.modulelist440 = ModuleList44()
        self.modulelist450 = ModuleList45()
        self.modulelist460 = ModuleList46()
    def forward(self, x0):
        x1 = self.modulelist380(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist390(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist400(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist410(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist420(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist430(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist440(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist450(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist460(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock1(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock1, self).__init__()
        self.modulelist200 = ModuleList20()
        self.modulelist210 = ModuleList21()
        self.modulelist220 = ModuleList22()
        self.modulelist230 = ModuleList23()
        self.modulelist240 = ModuleList24()
        self.modulelist250 = ModuleList25()
        self.modulelist260 = ModuleList26()
        self.modulelist270 = ModuleList27()
        self.modulelist280 = ModuleList28()
    def forward(self, x0):
        x1 = self.modulelist200(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist210(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist220(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist230(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist240(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist250(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist260(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist270(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist280(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock2(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock2, self).__init__()
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

class HarDBlock3(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock3, self).__init__()
        self.modulelist290 = ModuleList29()
        self.modulelist300 = ModuleList30()
        self.modulelist310 = ModuleList31()
        self.modulelist320 = ModuleList32()
        self.modulelist330 = ModuleList33()
        self.modulelist340 = ModuleList34()
        self.modulelist350 = ModuleList35()
        self.modulelist360 = ModuleList36()
        self.modulelist370 = ModuleList37()
    def forward(self, x0):
        x1 = self.modulelist290(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist300(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist310(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist320(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist330(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist340(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist350(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist360(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist370(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock4(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock4, self).__init__()
        self.modulelist470 = ModuleList47()
        self.modulelist480 = ModuleList48()
        self.modulelist490 = ModuleList49()
    def forward(self, x0):
        x1 = self.modulelist470(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist480(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8 = self.modulelist490(x7)
        x9 = [x1, x5, x8]
        x10 = paddle.concat(x=x9, axis=1)
        return x10

class Base_16(paddle.nn.Layer):
    def __init__(self, ):
        super(Base_16, self).__init__()
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
    def __init__(self, class_dim=1000):
        super(ModuleList, self).__init__()
        self.convlayer6_00 = ConvLayer6_0()
        self.convlayer1_00 = ConvLayer1_0()
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], ceil_mode=False)
        self.hardblock20 = HarDBlock2()
        self.convlayer4_00 = ConvLayer4_0()
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock10 = HarDBlock1()
        self.convlayer2_00 = ConvLayer2_0()
        self.hardblock30 = HarDBlock3()
        self.convlayer5_00 = ConvLayer5_0()
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock00 = HarDBlock0()
        self.convlayer3_00 = ConvLayer3_0()
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock40 = HarDBlock4()
        self.convlayer0_00 = ConvLayer0_0()
        self.base_160 = Base_16()
    def forward(self, x0):
        x1 = self.convlayer6_00(x0)
        x2 = self.convlayer1_00(x1)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x4 = self.pool0(x2)
        x5 = self.hardblock20(x4)
        x6 = self.convlayer4_00(x5)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x8 = self.pool1(x6)
        x9 = self.hardblock10(x8)
        x10 = self.convlayer2_00(x9)
        x11 = self.hardblock30(x10)
        x12 = self.convlayer5_00(x11)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x14 = self.pool2(x12)
        x15 = self.hardblock00(x14)
        x16 = self.convlayer3_00(x15)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x18 = self.pool3(x16)
        x19 = self.hardblock40(x18)
        x20 = self.convlayer0_00(x19)
        x21 = self.base_160(x20)
        return x21

class HarDNet68(paddle.nn.Layer):
    def __init__(self,class_dim=1000):
        super(HarDNet68, self).__init__()
        self.modulelist0 = ModuleList()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.modulelist0(x0)
        return x1

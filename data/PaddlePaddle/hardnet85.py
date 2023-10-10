import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import math
class ConvLayer0(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=24, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=24, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=28, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=28, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer2(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=256, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer3(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=36, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=36, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=24, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=24, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer6(paddle.nn.Layer):
    def __init__(self, conv0_in_channels):
        super(ConvLayer6, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=48, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=48, momentum=0.1, epsilon=1e-05)
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
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer8, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
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

class ConvLayer10(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer10, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer11(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels):
        super(ConvLayer11, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=conv0_out_channels, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=bn0_num_channels, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ModuleList18(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList18, self).__init__()
        self.convlayer00 = ConvLayer0(conv0_in_channels=96)
    def forward(self, x0):
        x1 = self.convlayer00(x0)
        return x1

class ModuleList19(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList19, self).__init__()
        self.convlayer80 = ConvLayer8(conv0_out_channels=40, conv0_in_channels=120, bn0_num_channels=40)
        self.convlayer00 = ConvLayer0(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer80(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList20(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList20, self).__init__()
        self.convlayer80 = ConvLayer8(conv0_out_channels=70, conv0_in_channels=160, bn0_num_channels=70)
        self.convlayer00 = ConvLayer0(conv0_in_channels=70)
    def forward(self, x0):
        x1 = self.convlayer80(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList21(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList21, self).__init__()
        self.convlayer80 = ConvLayer8(conv0_out_channels=40, conv0_in_channels=94, bn0_num_channels=40)
        self.convlayer00 = ConvLayer0(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer80(x0)
        x2 = self.convlayer00(x1)
        return x1, x2

class ModuleList22(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList22, self).__init__()
        self.convlayer80 = ConvLayer8(conv0_out_channels=118, conv0_in_channels=230, bn0_num_channels=118)
    def forward(self, x0):
        x1 = self.convlayer80(x0)
        return x1

class ModuleList23(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList23, self).__init__()
        self.convlayer50 = ConvLayer5(conv0_in_channels=192)
    def forward(self, x0):
        x1 = self.convlayer50(x0)
        return x1

class ModuleList24(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList24, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=40, conv0_in_channels=216, bn0_num_channels=40)
        self.convlayer50 = ConvLayer5(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList25(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList25, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=70, conv0_in_channels=256, bn0_num_channels=70)
        self.convlayer50 = ConvLayer5(conv0_in_channels=70)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList26(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList26, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=40, conv0_in_channels=94, bn0_num_channels=40)
        self.convlayer50 = ConvLayer5(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList27(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList27, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=118, conv0_in_channels=326, bn0_num_channels=118)
        self.convlayer50 = ConvLayer5(conv0_in_channels=118)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList28(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList28, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=40, conv0_in_channels=142, bn0_num_channels=40)
        self.convlayer50 = ConvLayer5(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList29(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList29, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=70, conv0_in_channels=182, bn0_num_channels=70)
        self.convlayer50 = ConvLayer5(conv0_in_channels=70)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList30(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList30, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=40, conv0_in_channels=94, bn0_num_channels=40)
        self.convlayer50 = ConvLayer5(conv0_in_channels=40)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        x2 = self.convlayer50(x1)
        return x1, x2

class ModuleList31(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList31, self).__init__()
        self.convlayer90 = ConvLayer9(conv0_out_channels=200, conv0_in_channels=444, bn0_num_channels=200)
    def forward(self, x0):
        x1 = self.convlayer90(x0)
        return x1

class ModuleList32(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList32, self).__init__()
        self.convlayer10 = ConvLayer1(conv0_in_channels=256)
    def forward(self, x0):
        x1 = self.convlayer10(x0)
        return x1

class ModuleList33(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList33, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=48, conv0_in_channels=284, bn0_num_channels=48)
        self.convlayer10 = ConvLayer1(conv0_in_channels=48)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList34(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList34, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=80, conv0_in_channels=332, bn0_num_channels=80)
        self.convlayer10 = ConvLayer1(conv0_in_channels=80)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList35(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList35, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=48, conv0_in_channels=108, bn0_num_channels=48)
        self.convlayer10 = ConvLayer1(conv0_in_channels=48)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList36(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList36, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=138, conv0_in_channels=412, bn0_num_channels=138)
        self.convlayer10 = ConvLayer1(conv0_in_channels=138)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList37(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList37, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=48, conv0_in_channels=166, bn0_num_channels=48)
        self.convlayer10 = ConvLayer1(conv0_in_channels=48)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList38(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList38, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=80, conv0_in_channels=214, bn0_num_channels=80)
        self.convlayer10 = ConvLayer1(conv0_in_channels=80)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList39(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList39, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=48, conv0_in_channels=108, bn0_num_channels=48)
        self.convlayer10 = ConvLayer1(conv0_in_channels=48)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        x2 = self.convlayer10(x1)
        return x1, x2

class ModuleList40(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList40, self).__init__()
        self.convlayer100 = ConvLayer10(conv0_out_channels=234, conv0_in_channels=550, bn0_num_channels=234)
    def forward(self, x0):
        x1 = self.convlayer100(x0)
        return x1

class ModuleList41(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList41, self).__init__()
        self.convlayer40 = ConvLayer4(conv0_in_channels=320)
    def forward(self, x0):
        x1 = self.convlayer40(x0)
        return x1

class ModuleList42(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList42, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=62, conv0_in_channels=356, bn0_num_channels=62)
        self.convlayer40 = ConvLayer4(conv0_in_channels=62)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList43(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList43, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=104, conv0_in_channels=418, bn0_num_channels=104)
        self.convlayer40 = ConvLayer4(conv0_in_channels=104)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList44(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList44, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=62, conv0_in_channels=140, bn0_num_channels=62)
        self.convlayer40 = ConvLayer4(conv0_in_channels=62)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList45(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList45, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=176, conv0_in_channels=522, bn0_num_channels=176)
        self.convlayer40 = ConvLayer4(conv0_in_channels=176)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList46(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList46, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=62, conv0_in_channels=212, bn0_num_channels=62)
        self.convlayer40 = ConvLayer4(conv0_in_channels=62)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList47(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList47, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=104, conv0_in_channels=274, bn0_num_channels=104)
        self.convlayer40 = ConvLayer4(conv0_in_channels=104)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList48(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList48, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=62, conv0_in_channels=140, bn0_num_channels=62)
        self.convlayer40 = ConvLayer4(conv0_in_channels=62)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        x2 = self.convlayer40(x1)
        return x1, x2

class ModuleList49(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList49, self).__init__()
        self.convlayer70 = ConvLayer7(conv0_out_channels=300, conv0_in_channels=698, bn0_num_channels=300)
    def forward(self, x0):
        x1 = self.convlayer70(x0)
        return x1

class ModuleList50(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList50, self).__init__()
        self.convlayer60 = ConvLayer6(conv0_in_channels=480)
    def forward(self, x0):
        x1 = self.convlayer60(x0)
        return x1

class ModuleList51(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList51, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=82, conv0_in_channels=528, bn0_num_channels=82)
        self.convlayer60 = ConvLayer6(conv0_in_channels=82)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList52(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList52, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=138, conv0_in_channels=610, bn0_num_channels=138)
        self.convlayer60 = ConvLayer6(conv0_in_channels=138)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList53(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList53, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=82, conv0_in_channels=186, bn0_num_channels=82)
        self.convlayer60 = ConvLayer6(conv0_in_channels=82)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList54(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList54, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=236, conv0_in_channels=748, bn0_num_channels=236)
        self.convlayer60 = ConvLayer6(conv0_in_channels=236)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList55(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList55, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=82, conv0_in_channels=284, bn0_num_channels=82)
        self.convlayer60 = ConvLayer6(conv0_in_channels=82)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList56(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList56, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=138, conv0_in_channels=366, bn0_num_channels=138)
        self.convlayer60 = ConvLayer6(conv0_in_channels=138)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList57(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList57, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=82, conv0_in_channels=186, bn0_num_channels=82)
        self.convlayer60 = ConvLayer6(conv0_in_channels=82)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        x2 = self.convlayer60(x1)
        return x1, x2

class ModuleList58(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList58, self).__init__()
        self.convlayer30 = ConvLayer3(conv0_out_channels=400, conv0_in_channels=984, bn0_num_channels=400)
    def forward(self, x0):
        x1 = self.convlayer30(x0)
        return x1

class ModuleList59(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList59, self).__init__()
        self.convlayer20 = ConvLayer2(conv0_in_channels=720)
    def forward(self, x0):
        x1 = self.convlayer20(x0)
        return x1

class ModuleList60(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList60, self).__init__()
        self.convlayer110 = ConvLayer11(conv0_out_channels=436, conv0_in_channels=976, bn0_num_channels=436)
        self.convlayer20 = ConvLayer2(conv0_in_channels=436)
    def forward(self, x0):
        x1 = self.convlayer110(x0)
        x2 = self.convlayer20(x1)
        return x1, x2

class ModuleList61(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList61, self).__init__()
        self.convlayer110 = ConvLayer11(conv0_out_channels=740, conv0_in_channels=1412, bn0_num_channels=740)
    def forward(self, x0):
        x1 = self.convlayer110(x0)
        return x1

class ConvLayer0_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer0_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=320, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=458)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=320, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer1_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer1_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1280, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=1252)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=1280, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer2_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer2_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=392)
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
        self.conv0 = paddle.nn.Conv2D(out_channels=48, kernel_size=(3, 3), bias_attr=False, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=48, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer4_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer4_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=96, kernel_size=(3, 3), bias_attr=False, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, in_channels=48)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=96, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer5_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer5_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=720, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=784)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=720, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer6_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer6_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=192, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=214)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=192, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class ConvLayer7_0(paddle.nn.Layer):
    def __init__(self, ):
        super(ConvLayer7_0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=480, kernel_size=(1, 1), bias_attr=False, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, in_channels=588)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=480, momentum=0.1, epsilon=1e-05)
        self.tanh0 = paddle.nn.Hardtanh(min=0.0, max=6.0)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.tanh0(x2)
        return x3

class HarDBlock0(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock0, self).__init__()
        self.modulelist180 = ModuleList18()
        self.modulelist190 = ModuleList19()
        self.modulelist200 = ModuleList20()
        self.modulelist210 = ModuleList21()
        self.modulelist220 = ModuleList22()
    def forward(self, x0):
        x1 = self.modulelist180(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist190(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist200(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist210(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16 = self.modulelist220(x15)
        x17 = [x1, x5, x9, x13, x16]
        x18 = paddle.concat(x=x17, axis=1)
        return x18

class HarDBlock1(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock1, self).__init__()
        self.modulelist500 = ModuleList50()
        self.modulelist510 = ModuleList51()
        self.modulelist520 = ModuleList52()
        self.modulelist530 = ModuleList53()
        self.modulelist540 = ModuleList54()
        self.modulelist550 = ModuleList55()
        self.modulelist560 = ModuleList56()
        self.modulelist570 = ModuleList57()
        self.modulelist580 = ModuleList58()
    def forward(self, x0):
        x1 = self.modulelist500(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist510(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist520(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist530(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist540(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist550(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist560(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist570(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist580(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock2(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock2, self).__init__()
        self.modulelist320 = ModuleList32()
        self.modulelist330 = ModuleList33()
        self.modulelist340 = ModuleList34()
        self.modulelist350 = ModuleList35()
        self.modulelist360 = ModuleList36()
        self.modulelist370 = ModuleList37()
        self.modulelist380 = ModuleList38()
        self.modulelist390 = ModuleList39()
        self.modulelist400 = ModuleList40()
    def forward(self, x0):
        x1 = self.modulelist320(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist330(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist340(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist350(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist360(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist370(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist380(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist390(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist400(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock3(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock3, self).__init__()
        self.modulelist230 = ModuleList23()
        self.modulelist240 = ModuleList24()
        self.modulelist250 = ModuleList25()
        self.modulelist260 = ModuleList26()
        self.modulelist270 = ModuleList27()
        self.modulelist280 = ModuleList28()
        self.modulelist290 = ModuleList29()
        self.modulelist300 = ModuleList30()
        self.modulelist310 = ModuleList31()
    def forward(self, x0):
        x1 = self.modulelist230(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist240(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist250(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist260(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist270(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist280(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist290(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist300(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist310(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock4(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock4, self).__init__()
        self.modulelist410 = ModuleList41()
        self.modulelist420 = ModuleList42()
        self.modulelist430 = ModuleList43()
        self.modulelist440 = ModuleList44()
        self.modulelist450 = ModuleList45()
        self.modulelist460 = ModuleList46()
        self.modulelist470 = ModuleList47()
        self.modulelist480 = ModuleList48()
        self.modulelist490 = ModuleList49()
    def forward(self, x0):
        x1 = self.modulelist410(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist420(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8,x9 = self.modulelist430(x7)
        x10 = [x9, x8]
        x11 = paddle.concat(x=x10, axis=1)
        x12,x13 = self.modulelist440(x11)
        x14 = [x13, x12, x8, x0]
        x15 = paddle.concat(x=x14, axis=1)
        x16,x17 = self.modulelist450(x15)
        x18 = [x17, x16]
        x19 = paddle.concat(x=x18, axis=1)
        x20,x21 = self.modulelist460(x19)
        x22 = [x21, x20, x16]
        x23 = paddle.concat(x=x22, axis=1)
        x24,x25 = self.modulelist470(x23)
        x26 = [x25, x24]
        x27 = paddle.concat(x=x26, axis=1)
        x28,x29 = self.modulelist480(x27)
        x30 = [x29, x28, x24, x16, x0]
        x31 = paddle.concat(x=x30, axis=1)
        x32 = self.modulelist490(x31)
        x33 = [x1, x5, x9, x13, x17, x21, x25, x29, x32]
        x34 = paddle.concat(x=x33, axis=1)
        return x34

class HarDBlock5(paddle.nn.Layer):
    def __init__(self, ):
        super(HarDBlock5, self).__init__()
        self.modulelist590 = ModuleList59()
        self.modulelist600 = ModuleList60()
        self.modulelist610 = ModuleList61()
    def forward(self, x0):
        x1 = self.modulelist590(x0)
        x2 = [x1, x0]
        x3 = paddle.concat(x=x2, axis=1)
        x4,x5 = self.modulelist600(x3)
        x6 = [x5, x4, x0]
        x7 = paddle.concat(x=x6, axis=1)
        x8 = self.modulelist610(x7)
        x9 = [x1, x5, x8]
        x10 = paddle.concat(x=x9, axis=1)
        return x10

class Base_19(paddle.nn.Layer):
    def __init__(self, ):
        super(Base_19, self).__init__()
        self.x1 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.linear0 = paddle.nn.Linear(in_features=1280, out_features=1000)
    def forward(self, x0):
        x2 = self.x1(x0)
        x3 = paddle.reshape(x=x2, shape=[x2.shape[0], -1])#reshape
        x4 = self.dropout0(x3)
        x5 = self.linear0(x4)
        return x5

class ModuleList(paddle.nn.Layer):
    def __init__(self, ):
        super(ModuleList, self).__init__()
        self.convlayer3_00 = ConvLayer3_0()
        self.convlayer4_00 = ConvLayer4_0()
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], ceil_mode=False)
        self.hardblock00 = HarDBlock0()
        self.convlayer6_00 = ConvLayer6_0()
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock30 = HarDBlock3()
        self.convlayer2_00 = ConvLayer2_0()
        self.hardblock20 = HarDBlock2()
        self.convlayer0_00 = ConvLayer0_0()
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock40 = HarDBlock4()
        self.convlayer7_00 = ConvLayer7_0()
        self.hardblock10 = HarDBlock1()
        self.convlayer5_00 = ConvLayer5_0()
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.hardblock50 = HarDBlock5()
        self.dropout0 = paddle.nn.Dropout(p=0.0)
        self.convlayer1_00 = ConvLayer1_0()
        self.base_190 = Base_19()
    def forward(self, x0):
        x1 = self.convlayer3_00(x0)
        x2 = self.convlayer4_00(x1)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x4 = self.pool0(x2)
        x5 = self.hardblock00(x4)
        x6 = self.convlayer6_00(x5)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x8 = self.pool1(x6)
        x9 = self.hardblock30(x8)
        x10 = self.convlayer2_00(x9)
        x11 = self.hardblock20(x10)
        x12 = self.convlayer0_00(x11)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x14 = self.pool2(x12)
        x15 = self.hardblock40(x14)
        x16 = self.convlayer7_00(x15)
        x17 = self.hardblock10(x16)
        x18 = self.convlayer5_00(x17)
        assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'
        x20 = self.pool3(x18)
        x21 = self.hardblock50(x20)
        x22 = self.dropout0(x21)
        x23 = self.convlayer1_00(x22)
        x24 = self.base_190(x23)
        return x24

class HarDNet85(paddle.nn.Layer):
    def __init__(self, class_dim=1000):
        super(HarDNet85, self).__init__()
        self.modulelist0 = ModuleList()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.modulelist0(x0)
        return x1

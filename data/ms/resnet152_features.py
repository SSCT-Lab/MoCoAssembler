import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_resnet152_path = "../imagenet_pretrained_weight/resnet152.ckpt"
class Module3(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module3, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module32(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module3_0_conv2d_0_in_channels,
                 module3_0_conv2d_0_out_channels, module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride,
                 module3_0_conv2d_0_padding, module3_0_conv2d_0_pad_mode, module3_1_conv2d_0_in_channels,
                 module3_1_conv2d_0_out_channels, module3_1_conv2d_0_kernel_size, module3_1_conv2d_0_stride,
                 module3_1_conv2d_0_padding, module3_1_conv2d_0_pad_mode):
        super(Module32, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode)
        self.module3_1 = Module3(conv2d_0_in_channels=module3_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_1_conv2d_0_stride,
                                 conv2d_0_padding=module3_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        opt_conv2d_0 = self.conv2d_0(module3_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(opt_conv2d_4, x)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module29(nn.Cell):

    def __init__(self):
        super(Module29, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256)
        self.module0_1 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module152(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_4_in_channels, module0_3_conv2d_4_out_channels,
                 module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels, module0_4_conv2d_2_in_channels,
                 module0_4_conv2d_2_out_channels, module0_4_conv2d_4_in_channels, module0_4_conv2d_4_out_channels,
                 module0_5_conv2d_0_in_channels, module0_5_conv2d_0_out_channels, module0_5_conv2d_2_in_channels,
                 module0_5_conv2d_2_out_channels, module0_5_conv2d_4_in_channels, module0_5_conv2d_4_out_channels,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_conv2d_4_in_channels, module0_6_conv2d_4_out_channels):
        super(Module152, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_4_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_4_conv2d_4_out_channels)
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_5_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_5_conv2d_4_out_channels)
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_6_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_6_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        return module0_6_opt


class Module407(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_4_in_channels, module0_3_conv2d_4_out_channels,
                 module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels, module0_4_conv2d_2_in_channels,
                 module0_4_conv2d_2_out_channels, module0_4_conv2d_4_in_channels, module0_4_conv2d_4_out_channels,
                 module0_5_conv2d_0_in_channels, module0_5_conv2d_0_out_channels, module0_5_conv2d_2_in_channels,
                 module0_5_conv2d_2_out_channels, module0_5_conv2d_4_in_channels, module0_5_conv2d_4_out_channels,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_conv2d_4_in_channels, module0_6_conv2d_4_out_channels,
                 module0_7_conv2d_0_in_channels, module0_7_conv2d_0_out_channels, module0_7_conv2d_2_in_channels,
                 module0_7_conv2d_2_out_channels, module0_7_conv2d_4_in_channels, module0_7_conv2d_4_out_channels):
        super(Module407, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_4_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_4_conv2d_4_out_channels)
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_5_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_5_conv2d_4_out_channels)
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_6_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_6_conv2d_4_out_channels)
        self.module0_7 = Module0(conv2d_0_in_channels=module0_7_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_7_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_7_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_7_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_7_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        return module0_7_opt


class Module149(nn.Cell):

    def __init__(self):
        super(Module149, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_1 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_2 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class ResNet_features(nn.Cell):

    def __init__(self):
        super(ResNet_features, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module32_0 = Module32(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   module3_0_conv2d_0_in_channels=64,
                                   module3_0_conv2d_0_out_channels=64,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=64,
                                   module3_1_conv2d_0_out_channels=64,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_4 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_10 = nn.ReLU()
        self.module29_0 = Module29()
        self.module32_1 = Module32(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module3_0_conv2d_0_in_channels=256,
                                   module3_0_conv2d_0_out_channels=128,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=128,
                                   module3_1_conv2d_0_out_channels=128,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_26 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_32 = nn.ReLU()
        self.module152_0 = Module152(module0_0_conv2d_0_in_channels=512,
                                     module0_0_conv2d_0_out_channels=128,
                                     module0_0_conv2d_2_in_channels=128,
                                     module0_0_conv2d_2_out_channels=128,
                                     module0_0_conv2d_4_in_channels=128,
                                     module0_0_conv2d_4_out_channels=512,
                                     module0_1_conv2d_0_in_channels=512,
                                     module0_1_conv2d_0_out_channels=128,
                                     module0_1_conv2d_2_in_channels=128,
                                     module0_1_conv2d_2_out_channels=128,
                                     module0_1_conv2d_4_in_channels=128,
                                     module0_1_conv2d_4_out_channels=512,
                                     module0_2_conv2d_0_in_channels=512,
                                     module0_2_conv2d_0_out_channels=128,
                                     module0_2_conv2d_2_in_channels=128,
                                     module0_2_conv2d_2_out_channels=128,
                                     module0_2_conv2d_4_in_channels=128,
                                     module0_2_conv2d_4_out_channels=512,
                                     module0_3_conv2d_0_in_channels=512,
                                     module0_3_conv2d_0_out_channels=128,
                                     module0_3_conv2d_2_in_channels=128,
                                     module0_3_conv2d_2_out_channels=128,
                                     module0_3_conv2d_4_in_channels=128,
                                     module0_3_conv2d_4_out_channels=512,
                                     module0_4_conv2d_0_in_channels=512,
                                     module0_4_conv2d_0_out_channels=128,
                                     module0_4_conv2d_2_in_channels=128,
                                     module0_4_conv2d_2_out_channels=128,
                                     module0_4_conv2d_4_in_channels=128,
                                     module0_4_conv2d_4_out_channels=512,
                                     module0_5_conv2d_0_in_channels=512,
                                     module0_5_conv2d_0_out_channels=128,
                                     module0_5_conv2d_2_in_channels=128,
                                     module0_5_conv2d_2_out_channels=128,
                                     module0_5_conv2d_4_in_channels=128,
                                     module0_5_conv2d_4_out_channels=512,
                                     module0_6_conv2d_0_in_channels=512,
                                     module0_6_conv2d_0_out_channels=128,
                                     module0_6_conv2d_2_in_channels=128,
                                     module0_6_conv2d_2_out_channels=128,
                                     module0_6_conv2d_4_in_channels=128,
                                     module0_6_conv2d_4_out_channels=512)
        self.module32_2 = Module32(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   module3_0_conv2d_0_in_channels=512,
                                   module3_0_conv2d_0_out_channels=256,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=256,
                                   module3_1_conv2d_0_out_channels=256,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_83 = nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_89 = nn.ReLU()
        self.module407_0 = Module407(module0_0_conv2d_0_in_channels=1024,
                                     module0_0_conv2d_0_out_channels=256,
                                     module0_0_conv2d_2_in_channels=256,
                                     module0_0_conv2d_2_out_channels=256,
                                     module0_0_conv2d_4_in_channels=256,
                                     module0_0_conv2d_4_out_channels=1024,
                                     module0_1_conv2d_0_in_channels=1024,
                                     module0_1_conv2d_0_out_channels=256,
                                     module0_1_conv2d_2_in_channels=256,
                                     module0_1_conv2d_2_out_channels=256,
                                     module0_1_conv2d_4_in_channels=256,
                                     module0_1_conv2d_4_out_channels=1024,
                                     module0_2_conv2d_0_in_channels=1024,
                                     module0_2_conv2d_0_out_channels=256,
                                     module0_2_conv2d_2_in_channels=256,
                                     module0_2_conv2d_2_out_channels=256,
                                     module0_2_conv2d_4_in_channels=256,
                                     module0_2_conv2d_4_out_channels=1024,
                                     module0_3_conv2d_0_in_channels=1024,
                                     module0_3_conv2d_0_out_channels=256,
                                     module0_3_conv2d_2_in_channels=256,
                                     module0_3_conv2d_2_out_channels=256,
                                     module0_3_conv2d_4_in_channels=256,
                                     module0_3_conv2d_4_out_channels=1024,
                                     module0_4_conv2d_0_in_channels=1024,
                                     module0_4_conv2d_0_out_channels=256,
                                     module0_4_conv2d_2_in_channels=256,
                                     module0_4_conv2d_2_out_channels=256,
                                     module0_4_conv2d_4_in_channels=256,
                                     module0_4_conv2d_4_out_channels=1024,
                                     module0_5_conv2d_0_in_channels=1024,
                                     module0_5_conv2d_0_out_channels=256,
                                     module0_5_conv2d_2_in_channels=256,
                                     module0_5_conv2d_2_out_channels=256,
                                     module0_5_conv2d_4_in_channels=256,
                                     module0_5_conv2d_4_out_channels=1024,
                                     module0_6_conv2d_0_in_channels=1024,
                                     module0_6_conv2d_0_out_channels=256,
                                     module0_6_conv2d_2_in_channels=256,
                                     module0_6_conv2d_2_out_channels=256,
                                     module0_6_conv2d_4_in_channels=256,
                                     module0_6_conv2d_4_out_channels=1024,
                                     module0_7_conv2d_0_in_channels=1024,
                                     module0_7_conv2d_0_out_channels=256,
                                     module0_7_conv2d_2_in_channels=256,
                                     module0_7_conv2d_2_out_channels=256,
                                     module0_7_conv2d_4_in_channels=256,
                                     module0_7_conv2d_4_out_channels=1024)
        self.module407_1 = Module407(module0_0_conv2d_0_in_channels=1024,
                                     module0_0_conv2d_0_out_channels=256,
                                     module0_0_conv2d_2_in_channels=256,
                                     module0_0_conv2d_2_out_channels=256,
                                     module0_0_conv2d_4_in_channels=256,
                                     module0_0_conv2d_4_out_channels=1024,
                                     module0_1_conv2d_0_in_channels=1024,
                                     module0_1_conv2d_0_out_channels=256,
                                     module0_1_conv2d_2_in_channels=256,
                                     module0_1_conv2d_2_out_channels=256,
                                     module0_1_conv2d_4_in_channels=256,
                                     module0_1_conv2d_4_out_channels=1024,
                                     module0_2_conv2d_0_in_channels=1024,
                                     module0_2_conv2d_0_out_channels=256,
                                     module0_2_conv2d_2_in_channels=256,
                                     module0_2_conv2d_2_out_channels=256,
                                     module0_2_conv2d_4_in_channels=256,
                                     module0_2_conv2d_4_out_channels=1024,
                                     module0_3_conv2d_0_in_channels=1024,
                                     module0_3_conv2d_0_out_channels=256,
                                     module0_3_conv2d_2_in_channels=256,
                                     module0_3_conv2d_2_out_channels=256,
                                     module0_3_conv2d_4_in_channels=256,
                                     module0_3_conv2d_4_out_channels=1024,
                                     module0_4_conv2d_0_in_channels=1024,
                                     module0_4_conv2d_0_out_channels=256,
                                     module0_4_conv2d_2_in_channels=256,
                                     module0_4_conv2d_2_out_channels=256,
                                     module0_4_conv2d_4_in_channels=256,
                                     module0_4_conv2d_4_out_channels=1024,
                                     module0_5_conv2d_0_in_channels=1024,
                                     module0_5_conv2d_0_out_channels=256,
                                     module0_5_conv2d_2_in_channels=256,
                                     module0_5_conv2d_2_out_channels=256,
                                     module0_5_conv2d_4_in_channels=256,
                                     module0_5_conv2d_4_out_channels=1024,
                                     module0_6_conv2d_0_in_channels=1024,
                                     module0_6_conv2d_0_out_channels=256,
                                     module0_6_conv2d_2_in_channels=256,
                                     module0_6_conv2d_2_out_channels=256,
                                     module0_6_conv2d_4_in_channels=256,
                                     module0_6_conv2d_4_out_channels=1024,
                                     module0_7_conv2d_0_in_channels=1024,
                                     module0_7_conv2d_0_out_channels=256,
                                     module0_7_conv2d_2_in_channels=256,
                                     module0_7_conv2d_2_out_channels=256,
                                     module0_7_conv2d_4_in_channels=256,
                                     module0_7_conv2d_4_out_channels=1024)
        self.module407_2 = Module407(module0_0_conv2d_0_in_channels=1024,
                                     module0_0_conv2d_0_out_channels=256,
                                     module0_0_conv2d_2_in_channels=256,
                                     module0_0_conv2d_2_out_channels=256,
                                     module0_0_conv2d_4_in_channels=256,
                                     module0_0_conv2d_4_out_channels=1024,
                                     module0_1_conv2d_0_in_channels=1024,
                                     module0_1_conv2d_0_out_channels=256,
                                     module0_1_conv2d_2_in_channels=256,
                                     module0_1_conv2d_2_out_channels=256,
                                     module0_1_conv2d_4_in_channels=256,
                                     module0_1_conv2d_4_out_channels=1024,
                                     module0_2_conv2d_0_in_channels=1024,
                                     module0_2_conv2d_0_out_channels=256,
                                     module0_2_conv2d_2_in_channels=256,
                                     module0_2_conv2d_2_out_channels=256,
                                     module0_2_conv2d_4_in_channels=256,
                                     module0_2_conv2d_4_out_channels=1024,
                                     module0_3_conv2d_0_in_channels=1024,
                                     module0_3_conv2d_0_out_channels=256,
                                     module0_3_conv2d_2_in_channels=256,
                                     module0_3_conv2d_2_out_channels=256,
                                     module0_3_conv2d_4_in_channels=256,
                                     module0_3_conv2d_4_out_channels=1024,
                                     module0_4_conv2d_0_in_channels=1024,
                                     module0_4_conv2d_0_out_channels=256,
                                     module0_4_conv2d_2_in_channels=256,
                                     module0_4_conv2d_2_out_channels=256,
                                     module0_4_conv2d_4_in_channels=256,
                                     module0_4_conv2d_4_out_channels=1024,
                                     module0_5_conv2d_0_in_channels=1024,
                                     module0_5_conv2d_0_out_channels=256,
                                     module0_5_conv2d_2_in_channels=256,
                                     module0_5_conv2d_2_out_channels=256,
                                     module0_5_conv2d_4_in_channels=256,
                                     module0_5_conv2d_4_out_channels=1024,
                                     module0_6_conv2d_0_in_channels=1024,
                                     module0_6_conv2d_0_out_channels=256,
                                     module0_6_conv2d_2_in_channels=256,
                                     module0_6_conv2d_2_out_channels=256,
                                     module0_6_conv2d_4_in_channels=256,
                                     module0_6_conv2d_4_out_channels=1024,
                                     module0_7_conv2d_0_in_channels=1024,
                                     module0_7_conv2d_0_out_channels=256,
                                     module0_7_conv2d_2_in_channels=256,
                                     module0_7_conv2d_2_out_channels=256,
                                     module0_7_conv2d_4_in_channels=256,
                                     module0_7_conv2d_4_out_channels=1024)
        self.module149_0 = Module149()
        self.module152_1 = Module152(module0_0_conv2d_0_in_channels=1024,
                                     module0_0_conv2d_0_out_channels=256,
                                     module0_0_conv2d_2_in_channels=256,
                                     module0_0_conv2d_2_out_channels=256,
                                     module0_0_conv2d_4_in_channels=256,
                                     module0_0_conv2d_4_out_channels=1024,
                                     module0_1_conv2d_0_in_channels=1024,
                                     module0_1_conv2d_0_out_channels=256,
                                     module0_1_conv2d_2_in_channels=256,
                                     module0_1_conv2d_2_out_channels=256,
                                     module0_1_conv2d_4_in_channels=256,
                                     module0_1_conv2d_4_out_channels=1024,
                                     module0_2_conv2d_0_in_channels=1024,
                                     module0_2_conv2d_0_out_channels=256,
                                     module0_2_conv2d_2_in_channels=256,
                                     module0_2_conv2d_2_out_channels=256,
                                     module0_2_conv2d_4_in_channels=256,
                                     module0_2_conv2d_4_out_channels=1024,
                                     module0_3_conv2d_0_in_channels=1024,
                                     module0_3_conv2d_0_out_channels=256,
                                     module0_3_conv2d_2_in_channels=256,
                                     module0_3_conv2d_2_out_channels=256,
                                     module0_3_conv2d_4_in_channels=256,
                                     module0_3_conv2d_4_out_channels=1024,
                                     module0_4_conv2d_0_in_channels=1024,
                                     module0_4_conv2d_0_out_channels=256,
                                     module0_4_conv2d_2_in_channels=256,
                                     module0_4_conv2d_2_out_channels=256,
                                     module0_4_conv2d_4_in_channels=256,
                                     module0_4_conv2d_4_out_channels=1024,
                                     module0_5_conv2d_0_in_channels=1024,
                                     module0_5_conv2d_0_out_channels=256,
                                     module0_5_conv2d_2_in_channels=256,
                                     module0_5_conv2d_2_out_channels=256,
                                     module0_5_conv2d_4_in_channels=256,
                                     module0_5_conv2d_4_out_channels=1024,
                                     module0_6_conv2d_0_in_channels=1024,
                                     module0_6_conv2d_0_out_channels=256,
                                     module0_6_conv2d_2_in_channels=256,
                                     module0_6_conv2d_2_out_channels=256,
                                     module0_6_conv2d_4_in_channels=256,
                                     module0_6_conv2d_4_out_channels=1024)
        self.module32_3 = Module32(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   module3_0_conv2d_0_in_channels=1024,
                                   module3_0_conv2d_0_out_channels=512,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=512,
                                   module3_1_conv2d_0_out_channels=512,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_336 = nn.Conv2d(in_channels=1024,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_342 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=2048)
        self.module32_4 = Module32(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   module3_0_conv2d_0_in_channels=2048,
                                   module3_0_conv2d_0_out_channels=512,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=512,
                                   module3_1_conv2d_0_out_channels=512,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.relu_356 = nn.ReLU()
        self.kernel_sizes = [7, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1]
        self.strides =  [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [3, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]


    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module32_0_opt = self.module32_0(opt_maxpool2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_maxpool2d_2)
        opt_add_9 = P.Add()(module32_0_opt, opt_conv2d_4)
        opt_relu_10 = self.relu_10(opt_add_9)
        module29_0_opt = self.module29_0(opt_relu_10)
        module32_1_opt = self.module32_1(module29_0_opt)
        opt_conv2d_26 = self.conv2d_26(module29_0_opt)
        opt_add_31 = P.Add()(module32_1_opt, opt_conv2d_26)
        opt_relu_32 = self.relu_32(opt_add_31)
        module152_0_opt = self.module152_0(opt_relu_32)
        module32_2_opt = self.module32_2(module152_0_opt)
        opt_conv2d_83 = self.conv2d_83(module152_0_opt)
        opt_add_88 = P.Add()(module32_2_opt, opt_conv2d_83)
        opt_relu_89 = self.relu_89(opt_add_88)
        module407_0_opt = self.module407_0(opt_relu_89)
        module407_1_opt = self.module407_1(module407_0_opt)
        module407_2_opt = self.module407_2(module407_1_opt)
        module149_0_opt = self.module149_0(module407_2_opt)
        module152_1_opt = self.module152_1(module149_0_opt)
        module32_3_opt = self.module32_3(module152_1_opt)
        opt_conv2d_336 = self.conv2d_336(module152_1_opt)
        opt_add_341 = P.Add()(module32_3_opt, opt_conv2d_336)
        opt_relu_342 = self.relu_342(opt_add_341)
        module0_0_opt = self.module0_0(opt_relu_342)
        module32_4_opt = self.module32_4(module0_0_opt)
        opt_add_355 = P.Add()(module32_4_opt, module0_0_opt)
        opt_relu_356 = self.relu_356(opt_add_355)
        return opt_relu_356
def resnet152_features():
    resnet152_model = ResNet_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_resnet152_path)
    mindspore.load_param_into_net(resnet152_model,param_dict)
    print("Load ResNet152 model pretrained from ImageNet")
    return resnet152_model
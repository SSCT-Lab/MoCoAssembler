import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_resnet50_path = "../imagenet_pretrained_weight/resnet50.ckpt"
class Module2(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module2, self).__init__()
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


class Module7(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module2_0_conv2d_0_in_channels,
                 module2_0_conv2d_0_out_channels, module2_0_conv2d_0_kernel_size, module2_0_conv2d_0_stride,
                 module2_0_conv2d_0_padding, module2_0_conv2d_0_pad_mode, module2_1_conv2d_0_in_channels,
                 module2_1_conv2d_0_out_channels, module2_1_conv2d_0_kernel_size, module2_1_conv2d_0_stride,
                 module2_1_conv2d_0_padding, module2_1_conv2d_0_pad_mode):
        super(Module7, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_0_conv2d_0_stride,
                                 conv2d_0_padding=module2_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_0_conv2d_0_pad_mode)
        self.module2_1 = Module2(conv2d_0_in_channels=module2_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_1_conv2d_0_stride,
                                 conv2d_0_padding=module2_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_1_conv2d_0_pad_mode)
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
        module2_0_opt = self.module2_0(x)
        module2_1_opt = self.module2_1(module2_0_opt)
        opt_conv2d_0 = self.conv2d_0(module2_1_opt)
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


class Module9(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module9, self).__init__()
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

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module12(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels):
        super(Module12, self).__init__()
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

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        return module0_2_opt


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
        self.module7_0 = Module7(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=256,
                                 module2_0_conv2d_0_in_channels=64,
                                 module2_0_conv2d_0_out_channels=64,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_1_conv2d_0_in_channels=64,
                                 module2_1_conv2d_0_out_channels=64,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(1, 1),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad")
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
        self.module9_0 = Module9(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=64,
                                 module0_0_conv2d_2_in_channels=64,
                                 module0_0_conv2d_2_out_channels=64,
                                 module0_0_conv2d_4_in_channels=64,
                                 module0_0_conv2d_4_out_channels=256,
                                 module0_1_conv2d_0_in_channels=256,
                                 module0_1_conv2d_0_out_channels=64,
                                 module0_1_conv2d_2_in_channels=64,
                                 module0_1_conv2d_2_out_channels=64,
                                 module0_1_conv2d_4_in_channels=64,
                                 module0_1_conv2d_4_out_channels=256)
        self.module7_1 = Module7(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_in_channels=256,
                                 module2_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_1_conv2d_0_in_channels=128,
                                 module2_1_conv2d_0_out_channels=128,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad")
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
        self.module12_0 = Module12(module0_0_conv2d_0_in_channels=512,
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
                                   module0_2_conv2d_4_out_channels=512)
        self.module7_2 = Module7(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=1024,
                                 module2_0_conv2d_0_in_channels=512,
                                 module2_0_conv2d_0_out_channels=256,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_1_conv2d_0_in_channels=256,
                                 module2_1_conv2d_0_out_channels=256,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad")
        self.conv2d_55 = nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_61 = nn.ReLU()
        self.module9_1 = Module9(module0_0_conv2d_0_in_channels=1024,
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
                                 module0_1_conv2d_4_out_channels=1024)
        self.module12_1 = Module12(module0_0_conv2d_0_in_channels=1024,
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
                                   module0_2_conv2d_4_out_channels=1024)
        self.module7_3 = Module7(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=2048,
                                 module2_0_conv2d_0_in_channels=1024,
                                 module2_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_1_conv2d_0_in_channels=512,
                                 module2_1_conv2d_0_out_channels=512,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad")
        self.conv2d_98 = nn.Conv2d(in_channels=1024,
                                   out_channels=2048,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_104 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=2048)
        self.module7_4 = Module7(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=2048,
                                 module2_0_conv2d_0_in_channels=2048,
                                 module2_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_1_conv2d_0_in_channels=512,
                                 module2_1_conv2d_0_out_channels=512,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(1, 1),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad")
        self.relu_118 = nn.ReLU()

        self.kernel_sizes = [7, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1]
        self.strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [3, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module7_0_opt = self.module7_0(opt_maxpool2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_maxpool2d_2)
        opt_add_9 = P.Add()(module7_0_opt, opt_conv2d_4)
        opt_relu_10 = self.relu_10(opt_add_9)
        module9_0_opt = self.module9_0(opt_relu_10)
        module7_1_opt = self.module7_1(module9_0_opt)
        opt_conv2d_26 = self.conv2d_26(module9_0_opt)
        opt_add_31 = P.Add()(module7_1_opt, opt_conv2d_26)
        opt_relu_32 = self.relu_32(opt_add_31)
        module12_0_opt = self.module12_0(opt_relu_32)
        module7_2_opt = self.module7_2(module12_0_opt)
        opt_conv2d_55 = self.conv2d_55(module12_0_opt)
        opt_add_60 = P.Add()(module7_2_opt, opt_conv2d_55)
        opt_relu_61 = self.relu_61(opt_add_60)
        module9_1_opt = self.module9_1(opt_relu_61)
        module12_1_opt = self.module12_1(module9_1_opt)
        module7_3_opt = self.module7_3(module12_1_opt)
        opt_conv2d_98 = self.conv2d_98(module12_1_opt)
        opt_add_103 = P.Add()(module7_3_opt, opt_conv2d_98)
        opt_relu_104 = self.relu_104(opt_add_103)
        module0_0_opt = self.module0_0(opt_relu_104)
        module7_4_opt = self.module7_4(module0_0_opt)
        opt_add_117 = P.Add()(module7_4_opt, module0_0_opt)
        opt_relu_118 = self.relu_118(opt_add_117)
        return opt_relu_118

def resnet50_features():
    resnet50_model = ResNet_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_resnet50_path)
    mindspore.load_param_into_net(resnet50_model,param_dict)
    print("Load ResNet50 model pretrained from ImageNet")
    return resnet50_model
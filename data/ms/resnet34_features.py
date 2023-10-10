import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_resnet34_path = "../imagenet_pretrained_weight/resnet34.ckpt"

class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_5_in_channels, conv2d_5_out_channels, conv2d_7_in_channels, conv2d_7_out_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()
        self.conv2d_7 = nn.Conv2d(in_channels=conv2d_7_in_channels,
                                  out_channels=conv2d_7_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_9 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_add_3 = P.Add()(opt_conv2d_2, x)
        opt_relu_4 = self.relu_4(opt_add_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_conv2d_7 = self.conv2d_7(opt_relu_6)
        opt_add_8 = P.Add()(opt_conv2d_7, opt_relu_4)
        opt_relu_9 = self.relu_9(opt_add_8)
        return opt_relu_9


class Module2(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride):
        super(Module2, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=conv2d_0_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module8(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_5_in_channels, module0_0_conv2d_5_out_channels, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels, module2_0_conv2d_0_in_channels, module2_0_conv2d_0_out_channels,
                 module2_0_conv2d_0_stride):
        super(Module8, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_5_in_channels=module0_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module0_0_conv2d_5_out_channels,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels)
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_stride=module2_0_conv2d_0_stride)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module2_0_opt = self.module2_0(module0_0_opt)
        opt_conv2d_0 = self.conv2d_0(module2_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, module0_0_opt)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module4(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module2_0_conv2d_0_in_channels,
                 module2_0_conv2d_0_out_channels, module2_0_conv2d_0_stride):
        super(Module4, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_stride=module2_0_conv2d_0_stride)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module2_0_opt = self.module2_0(x)
        opt_conv2d_0 = self.conv2d_0(module2_0_opt)
        return opt_conv2d_0


class Module6(nn.Cell):

    def __init__(self):
        super(Module6, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=512, conv2d_0_out_channels=512, conv2d_0_stride=(1, 1))
        self.conv2d_0 = nn.Conv2d(in_channels=512,
                                  out_channels=512,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()

    def construct(self, x):
        module2_0_opt = self.module2_0(x)
        opt_conv2d_0 = self.conv2d_0(module2_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, x)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


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
        self.module8_0 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 module0_0_conv2d_0_in_channels=64,
                                 module0_0_conv2d_0_out_channels=64,
                                 module0_0_conv2d_2_in_channels=64,
                                 module0_0_conv2d_2_out_channels=64,
                                 module0_0_conv2d_5_in_channels=64,
                                 module0_0_conv2d_5_out_channels=64,
                                 module0_0_conv2d_7_in_channels=64,
                                 module0_0_conv2d_7_out_channels=64,
                                 module2_0_conv2d_0_in_channels=64,
                                 module2_0_conv2d_0_out_channels=64,
                                 module2_0_conv2d_0_stride=(1, 1))
        self.module4_0 = Module4(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_in_channels=64,
                                 module2_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_stride=(2, 2))
        self.conv2d_19 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_23 = nn.ReLU()
        self.module8_1 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module0_0_conv2d_0_in_channels=128,
                                 module0_0_conv2d_0_out_channels=128,
                                 module0_0_conv2d_2_in_channels=128,
                                 module0_0_conv2d_2_out_channels=128,
                                 module0_0_conv2d_5_in_channels=128,
                                 module0_0_conv2d_5_out_channels=128,
                                 module0_0_conv2d_7_in_channels=128,
                                 module0_0_conv2d_7_out_channels=128,
                                 module2_0_conv2d_0_in_channels=128,
                                 module2_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_stride=(1, 1))
        self.module4_1 = Module4(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 module2_0_conv2d_0_in_channels=128,
                                 module2_0_conv2d_0_out_channels=256,
                                 module2_0_conv2d_0_stride=(2, 2))
        self.conv2d_40 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_44 = nn.ReLU()
        self.module0_0 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_5_in_channels=256,
                                 conv2d_5_out_channels=256,
                                 conv2d_7_in_channels=256,
                                 conv2d_7_out_channels=256)
        self.module8_2 = Module8(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_2_in_channels=256,
                                 module0_0_conv2d_2_out_channels=256,
                                 module0_0_conv2d_5_in_channels=256,
                                 module0_0_conv2d_5_out_channels=256,
                                 module0_0_conv2d_7_in_channels=256,
                                 module0_0_conv2d_7_out_channels=256,
                                 module2_0_conv2d_0_in_channels=256,
                                 module2_0_conv2d_0_out_channels=256,
                                 module2_0_conv2d_0_stride=(1, 1))
        self.module4_2 = Module4(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_in_channels=256,
                                 module2_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_stride=(2, 2))
        self.conv2d_71 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_75 = nn.ReLU()
        self.module6_0 = Module6()
        self.module4_3 = Module4(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_in_channels=512,
                                 module2_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_0_stride=(1, 1))
        self.relu_85 = nn.ReLU()

        self.kernel_sizes = [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        self.strides = [2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
        self.paddings = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module8_0_opt = self.module8_0(opt_maxpool2d_2)
        module4_0_opt = self.module4_0(module8_0_opt)
        opt_conv2d_19 = self.conv2d_19(module8_0_opt)
        opt_add_22 = P.Add()(module4_0_opt, opt_conv2d_19)
        opt_relu_23 = self.relu_23(opt_add_22)
        module8_1_opt = self.module8_1(opt_relu_23)
        module4_1_opt = self.module4_1(module8_1_opt)
        opt_conv2d_40 = self.conv2d_40(module8_1_opt)
        opt_add_43 = P.Add()(module4_1_opt, opt_conv2d_40)
        opt_relu_44 = self.relu_44(opt_add_43)
        module0_0_opt = self.module0_0(opt_relu_44)
        module8_2_opt = self.module8_2(module0_0_opt)
        module4_2_opt = self.module4_2(module8_2_opt)
        opt_conv2d_71 = self.conv2d_71(module8_2_opt)
        opt_add_74 = P.Add()(module4_2_opt, opt_conv2d_71)
        opt_relu_75 = self.relu_75(opt_add_74)
        module6_0_opt = self.module6_0(opt_relu_75)
        module4_3_opt = self.module4_3(module6_0_opt)
        opt_add_84 = P.Add()(module4_3_opt, module6_0_opt)
        opt_relu_85 = self.relu_85(opt_add_84)
        return opt_relu_85

def resnet34_features():
    resnet34_model = ResNet_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_resnet34_path)
    mindspore.load_param_into_net(resnet34_model,param_dict)
    print("Load ResNet34 model pretrained from ImageNet")
    return resnet34_model

if __name__  == "__main__":
    resnet34_features()
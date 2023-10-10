import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_densnet161_model_path = "../imagenet_pretrained_weight/densenet161.ckpt"
class Module0(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features, conv2d_2_in_channels):
        super(Module0, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=192,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=192,
                                  out_channels=48,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.concat_5 = P.Concat(axis=1)

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_concat_5 = self.concat_5((x, opt_conv2d_4, ))
        return opt_concat_5


class Module4(nn.Cell):

    def __init__(self, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
                 module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels,
                 module0_2_batchnorm2d_0_num_features, module0_2_conv2d_2_in_channels,
                 module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels):
        super(Module4, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class Module2(nn.Cell):

    def __init__(self):
        super(Module2, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=288, conv2d_2_in_channels=288)
        self.module0_1 = Module0(batchnorm2d_0_num_features=336, conv2d_2_in_channels=336)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module1330(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features):
        super(Module1330, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class Module222(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module222, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.pad_avgpool2d_1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_avgpool2d_1 = self.pad_avgpool2d_1(opt_conv2d_0)
        opt_avgpool2d_1 = self.avgpool2d_1(opt_avgpool2d_1)
        return opt_avgpool2d_1


class Module1332(nn.Cell):

    def __init__(self):
        super(Module1332, self).__init__()
        self.module1330_0 = Module1330(batchnorm2d_0_num_features=384)
        self.module222_0 = Module222(conv2d_0_in_channels=384, conv2d_0_out_channels=192)
        self.module0_0 = Module0(batchnorm2d_0_num_features=192, conv2d_2_in_channels=192)
        self.module0_1 = Module0(batchnorm2d_0_num_features=240, conv2d_2_in_channels=240)
        self.module0_2 = Module0(batchnorm2d_0_num_features=288, conv2d_2_in_channels=288)
        self.module0_3 = Module0(batchnorm2d_0_num_features=336, conv2d_2_in_channels=336)
        self.module0_4 = Module0(batchnorm2d_0_num_features=384, conv2d_2_in_channels=384)
        self.module0_5 = Module0(batchnorm2d_0_num_features=432, conv2d_2_in_channels=432)
        self.module0_6 = Module0(batchnorm2d_0_num_features=480, conv2d_2_in_channels=480)
        self.module0_7 = Module0(batchnorm2d_0_num_features=528, conv2d_2_in_channels=528)
        self.module0_8 = Module0(batchnorm2d_0_num_features=576, conv2d_2_in_channels=576)
        self.module0_9 = Module0(batchnorm2d_0_num_features=624, conv2d_2_in_channels=624)
        self.module0_10 = Module0(batchnorm2d_0_num_features=672, conv2d_2_in_channels=672)
        self.module0_11 = Module0(batchnorm2d_0_num_features=720, conv2d_2_in_channels=720)

    def construct(self, x):
        module1330_0_opt = self.module1330_0(x)
        module222_0_opt = self.module222_0(module1330_0_opt)
        module0_0_opt = self.module0_0(module222_0_opt)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        module0_8_opt = self.module0_8(module0_7_opt)
        module0_9_opt = self.module0_9(module0_8_opt)
        module0_10_opt = self.module0_10(module0_9_opt)
        module0_11_opt = self.module0_11(module0_10_opt)
        return module0_11_opt


class Module1334(nn.Cell):

    def __init__(
            self, module1330_0_batchnorm2d_0_num_features, module222_0_conv2d_0_in_channels,
            module222_0_conv2d_0_out_channels, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
            module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_2_batchnorm2d_0_num_features,
            module0_2_conv2d_2_in_channels, module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels,
            module0_4_batchnorm2d_0_num_features, module0_4_conv2d_2_in_channels, module0_5_batchnorm2d_0_num_features,
            module0_5_conv2d_2_in_channels, module0_6_batchnorm2d_0_num_features, module0_6_conv2d_2_in_channels,
            module0_7_batchnorm2d_0_num_features, module0_7_conv2d_2_in_channels, module0_8_batchnorm2d_0_num_features,
            module0_8_conv2d_2_in_channels, module0_9_batchnorm2d_0_num_features, module0_9_conv2d_2_in_channels,
            module0_10_batchnorm2d_0_num_features, module0_10_conv2d_2_in_channels,
            module0_11_batchnorm2d_0_num_features, module0_11_conv2d_2_in_channels,
            module0_12_batchnorm2d_0_num_features, module0_12_conv2d_2_in_channels,
            module0_13_batchnorm2d_0_num_features, module0_13_conv2d_2_in_channels,
            module0_14_batchnorm2d_0_num_features, module0_14_conv2d_2_in_channels,
            module0_15_batchnorm2d_0_num_features, module0_15_conv2d_2_in_channels,
            module0_16_batchnorm2d_0_num_features, module0_16_conv2d_2_in_channels,
            module0_17_batchnorm2d_0_num_features, module0_17_conv2d_2_in_channels,
            module0_18_batchnorm2d_0_num_features, module0_18_conv2d_2_in_channels,
            module0_19_batchnorm2d_0_num_features, module0_19_conv2d_2_in_channels):
        super(Module1334, self).__init__()
        self.module1330_0 = Module1330(batchnorm2d_0_num_features=module1330_0_batchnorm2d_0_num_features)
        self.module222_0 = Module222(conv2d_0_in_channels=module222_0_conv2d_0_in_channels,
                                     conv2d_0_out_channels=module222_0_conv2d_0_out_channels)
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)
        self.module0_4 = Module0(batchnorm2d_0_num_features=module0_4_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels)
        self.module0_5 = Module0(batchnorm2d_0_num_features=module0_5_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels)
        self.module0_6 = Module0(batchnorm2d_0_num_features=module0_6_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels)
        self.module0_7 = Module0(batchnorm2d_0_num_features=module0_7_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels)
        self.module0_8 = Module0(batchnorm2d_0_num_features=module0_8_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_8_conv2d_2_in_channels)
        self.module0_9 = Module0(batchnorm2d_0_num_features=module0_9_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_9_conv2d_2_in_channels)
        self.module0_10 = Module0(batchnorm2d_0_num_features=module0_10_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_10_conv2d_2_in_channels)
        self.module0_11 = Module0(batchnorm2d_0_num_features=module0_11_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_11_conv2d_2_in_channels)
        self.module0_12 = Module0(batchnorm2d_0_num_features=module0_12_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_12_conv2d_2_in_channels)
        self.module0_13 = Module0(batchnorm2d_0_num_features=module0_13_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_13_conv2d_2_in_channels)
        self.module0_14 = Module0(batchnorm2d_0_num_features=module0_14_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_14_conv2d_2_in_channels)
        self.module0_15 = Module0(batchnorm2d_0_num_features=module0_15_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_15_conv2d_2_in_channels)
        self.module0_16 = Module0(batchnorm2d_0_num_features=module0_16_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_16_conv2d_2_in_channels)
        self.module0_17 = Module0(batchnorm2d_0_num_features=module0_17_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_17_conv2d_2_in_channels)
        self.module0_18 = Module0(batchnorm2d_0_num_features=module0_18_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_18_conv2d_2_in_channels)
        self.module0_19 = Module0(batchnorm2d_0_num_features=module0_19_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_19_conv2d_2_in_channels)

    def construct(self, x):
        module1330_0_opt = self.module1330_0(x)
        module222_0_opt = self.module222_0(module1330_0_opt)
        module0_0_opt = self.module0_0(module222_0_opt)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        module0_8_opt = self.module0_8(module0_7_opt)
        module0_9_opt = self.module0_9(module0_8_opt)
        module0_10_opt = self.module0_10(module0_9_opt)
        module0_11_opt = self.module0_11(module0_10_opt)
        module0_12_opt = self.module0_12(module0_11_opt)
        module0_13_opt = self.module0_13(module0_12_opt)
        module0_14_opt = self.module0_14(module0_13_opt)
        module0_15_opt = self.module0_15(module0_14_opt)
        module0_16_opt = self.module0_16(module0_15_opt)
        module0_17_opt = self.module0_17(module0_16_opt)
        module0_18_opt = self.module0_18(module0_17_opt)
        module0_19_opt = self.module0_19(module0_18_opt)
        return module0_19_opt


class Module335(nn.Cell):

    def __init__(
            self, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
            module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_2_batchnorm2d_0_num_features,
            module0_2_conv2d_2_in_channels, module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels,
            module0_4_batchnorm2d_0_num_features, module0_4_conv2d_2_in_channels, module0_5_batchnorm2d_0_num_features,
            module0_5_conv2d_2_in_channels, module0_6_batchnorm2d_0_num_features, module0_6_conv2d_2_in_channels,
            module0_7_batchnorm2d_0_num_features, module0_7_conv2d_2_in_channels):
        super(Module335, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)
        self.module0_4 = Module0(batchnorm2d_0_num_features=module0_4_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels)
        self.module0_5 = Module0(batchnorm2d_0_num_features=module0_5_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels)
        self.module0_6 = Module0(batchnorm2d_0_num_features=module0_6_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels)
        self.module0_7 = Module0(batchnorm2d_0_num_features=module0_7_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels)

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


class DenseNet_features(nn.Cell):

    def __init__(self):
        super(DenseNet_features, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=96,
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
        self.module4_0 = Module4(module0_0_batchnorm2d_0_num_features=96,
                                 module0_0_conv2d_2_in_channels=96,
                                 module0_1_batchnorm2d_0_num_features=144,
                                 module0_1_conv2d_2_in_channels=144,
                                 module0_2_batchnorm2d_0_num_features=192,
                                 module0_2_conv2d_2_in_channels=192,
                                 module0_3_batchnorm2d_0_num_features=240,
                                 module0_3_conv2d_2_in_channels=240)
        self.module2_0 = Module2()
        self.module1332_0 = Module1332()
        self.module1334_0 = Module1334(module1330_0_batchnorm2d_0_num_features=768,
                                       module222_0_conv2d_0_in_channels=768,
                                       module222_0_conv2d_0_out_channels=384,
                                       module0_0_batchnorm2d_0_num_features=384,
                                       module0_0_conv2d_2_in_channels=384,
                                       module0_1_batchnorm2d_0_num_features=432,
                                       module0_1_conv2d_2_in_channels=432,
                                       module0_2_batchnorm2d_0_num_features=480,
                                       module0_2_conv2d_2_in_channels=480,
                                       module0_3_batchnorm2d_0_num_features=528,
                                       module0_3_conv2d_2_in_channels=528,
                                       module0_4_batchnorm2d_0_num_features=576,
                                       module0_4_conv2d_2_in_channels=576,
                                       module0_5_batchnorm2d_0_num_features=624,
                                       module0_5_conv2d_2_in_channels=624,
                                       module0_6_batchnorm2d_0_num_features=672,
                                       module0_6_conv2d_2_in_channels=672,
                                       module0_7_batchnorm2d_0_num_features=720,
                                       module0_7_conv2d_2_in_channels=720,
                                       module0_8_batchnorm2d_0_num_features=768,
                                       module0_8_conv2d_2_in_channels=768,
                                       module0_9_batchnorm2d_0_num_features=816,
                                       module0_9_conv2d_2_in_channels=816,
                                       module0_10_batchnorm2d_0_num_features=864,
                                       module0_10_conv2d_2_in_channels=864,
                                       module0_11_batchnorm2d_0_num_features=912,
                                       module0_11_conv2d_2_in_channels=912,
                                       module0_12_batchnorm2d_0_num_features=960,
                                       module0_12_conv2d_2_in_channels=960,
                                       module0_13_batchnorm2d_0_num_features=1008,
                                       module0_13_conv2d_2_in_channels=1008,
                                       module0_14_batchnorm2d_0_num_features=1056,
                                       module0_14_conv2d_2_in_channels=1056,
                                       module0_15_batchnorm2d_0_num_features=1104,
                                       module0_15_conv2d_2_in_channels=1104,
                                       module0_16_batchnorm2d_0_num_features=1152,
                                       module0_16_conv2d_2_in_channels=1152,
                                       module0_17_batchnorm2d_0_num_features=1200,
                                       module0_17_conv2d_2_in_channels=1200,
                                       module0_18_batchnorm2d_0_num_features=1248,
                                       module0_18_conv2d_2_in_channels=1248,
                                       module0_19_batchnorm2d_0_num_features=1296,
                                       module0_19_conv2d_2_in_channels=1296)
        self.module335_0 = Module335(module0_0_batchnorm2d_0_num_features=1344,
                                     module0_0_conv2d_2_in_channels=1344,
                                     module0_1_batchnorm2d_0_num_features=1392,
                                     module0_1_conv2d_2_in_channels=1392,
                                     module0_2_batchnorm2d_0_num_features=1440,
                                     module0_2_conv2d_2_in_channels=1440,
                                     module0_3_batchnorm2d_0_num_features=1488,
                                     module0_3_conv2d_2_in_channels=1488,
                                     module0_4_batchnorm2d_0_num_features=1536,
                                     module0_4_conv2d_2_in_channels=1536,
                                     module0_5_batchnorm2d_0_num_features=1584,
                                     module0_5_conv2d_2_in_channels=1584,
                                     module0_6_batchnorm2d_0_num_features=1632,
                                     module0_6_conv2d_2_in_channels=1632,
                                     module0_7_batchnorm2d_0_num_features=1680,
                                     module0_7_conv2d_2_in_channels=1680)
        self.module335_1 = Module335(module0_0_batchnorm2d_0_num_features=1728,
                                     module0_0_conv2d_2_in_channels=1728,
                                     module0_1_batchnorm2d_0_num_features=1776,
                                     module0_1_conv2d_2_in_channels=1776,
                                     module0_2_batchnorm2d_0_num_features=1824,
                                     module0_2_conv2d_2_in_channels=1824,
                                     module0_3_batchnorm2d_0_num_features=1872,
                                     module0_3_conv2d_2_in_channels=1872,
                                     module0_4_batchnorm2d_0_num_features=1920,
                                     module0_4_conv2d_2_in_channels=1920,
                                     module0_5_batchnorm2d_0_num_features=1968,
                                     module0_5_conv2d_2_in_channels=1968,
                                     module0_6_batchnorm2d_0_num_features=2016,
                                     module0_6_conv2d_2_in_channels=2016,
                                     module0_7_batchnorm2d_0_num_features=2064,
                                     module0_7_conv2d_2_in_channels=2064)
        self.module1334_1 = Module1334(module1330_0_batchnorm2d_0_num_features=2112,
                                       module222_0_conv2d_0_in_channels=2112,
                                       module222_0_conv2d_0_out_channels=1056,
                                       module0_0_batchnorm2d_0_num_features=1056,
                                       module0_0_conv2d_2_in_channels=1056,
                                       module0_1_batchnorm2d_0_num_features=1104,
                                       module0_1_conv2d_2_in_channels=1104,
                                       module0_2_batchnorm2d_0_num_features=1152,
                                       module0_2_conv2d_2_in_channels=1152,
                                       module0_3_batchnorm2d_0_num_features=1200,
                                       module0_3_conv2d_2_in_channels=1200,
                                       module0_4_batchnorm2d_0_num_features=1248,
                                       module0_4_conv2d_2_in_channels=1248,
                                       module0_5_batchnorm2d_0_num_features=1296,
                                       module0_5_conv2d_2_in_channels=1296,
                                       module0_6_batchnorm2d_0_num_features=1344,
                                       module0_6_conv2d_2_in_channels=1344,
                                       module0_7_batchnorm2d_0_num_features=1392,
                                       module0_7_conv2d_2_in_channels=1392,
                                       module0_8_batchnorm2d_0_num_features=1440,
                                       module0_8_conv2d_2_in_channels=1440,
                                       module0_9_batchnorm2d_0_num_features=1488,
                                       module0_9_conv2d_2_in_channels=1488,
                                       module0_10_batchnorm2d_0_num_features=1536,
                                       module0_10_conv2d_2_in_channels=1536,
                                       module0_11_batchnorm2d_0_num_features=1584,
                                       module0_11_conv2d_2_in_channels=1584,
                                       module0_12_batchnorm2d_0_num_features=1632,
                                       module0_12_conv2d_2_in_channels=1632,
                                       module0_13_batchnorm2d_0_num_features=1680,
                                       module0_13_conv2d_2_in_channels=1680,
                                       module0_14_batchnorm2d_0_num_features=1728,
                                       module0_14_conv2d_2_in_channels=1728,
                                       module0_15_batchnorm2d_0_num_features=1776,
                                       module0_15_conv2d_2_in_channels=1776,
                                       module0_16_batchnorm2d_0_num_features=1824,
                                       module0_16_conv2d_2_in_channels=1824,
                                       module0_17_batchnorm2d_0_num_features=1872,
                                       module0_17_conv2d_2_in_channels=1872,
                                       module0_18_batchnorm2d_0_num_features=1920,
                                       module0_18_conv2d_2_in_channels=1920,
                                       module0_19_batchnorm2d_0_num_features=1968,
                                       module0_19_conv2d_2_in_channels=1968)
        self.module4_1 = Module4(module0_0_batchnorm2d_0_num_features=2016,
                                 module0_0_conv2d_2_in_channels=2016,
                                 module0_1_batchnorm2d_0_num_features=2064,
                                 module0_1_conv2d_2_in_channels=2064,
                                 module0_2_batchnorm2d_0_num_features=2112,
                                 module0_2_conv2d_2_in_channels=2112,
                                 module0_3_batchnorm2d_0_num_features=2160,
                                 module0_3_conv2d_2_in_channels=2160)
        self.batchnorm2d_483 = nn.BatchNorm2d(num_features=2208, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_484 = nn.ReLU()
        self.kernel_sizes = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
        self.strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module4_0_opt = self.module4_0(opt_maxpool2d_2)
        module2_0_opt = self.module2_0(module4_0_opt)
        module1332_0_opt = self.module1332_0(module2_0_opt)
        module1334_0_opt = self.module1334_0(module1332_0_opt)
        module335_0_opt = self.module335_0(module1334_0_opt)
        module335_1_opt = self.module335_1(module335_0_opt)
        module1334_1_opt = self.module1334_1(module335_1_opt)
        module4_1_opt = self.module4_1(module1334_1_opt)
        opt_batchnorm2d_483 = self.batchnorm2d_483(module4_1_opt)
        opt_relu_484 = self.relu_484(opt_batchnorm2d_483)
        return opt_relu_484

def densnet161_features():
    densnet161_model = DenseNet_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_densnet161_model_path)
    mindspore.load_param_into_net(densnet161_model,param_dict)
    print("Load densnet161_model model pretrained from ImageNet")
    return densnet161_model

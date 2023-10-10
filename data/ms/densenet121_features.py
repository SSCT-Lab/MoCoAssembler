import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_densnet121_model_path = "../imagenet_pretrained_weight/densenet121.ckpt"

class Module0(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features, conv2d_2_in_channels):
        super(Module0, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=128,
                                  out_channels=32,
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


class Module19(nn.Cell):

    def __init__(self, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
                 module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels,
                 module0_2_batchnorm2d_0_num_features, module0_2_conv2d_2_in_channels,
                 module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels):
        super(Module19, self).__init__()
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


class Module43(nn.Cell):

    def __init__(self):
        super(Module43, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=192, conv2d_2_in_channels=192)
        self.module0_1 = Module0(batchnorm2d_0_num_features=224, conv2d_2_in_channels=224)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=256, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=256,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.pad_avgpool2d_3 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_2 = Module0(batchnorm2d_0_num_features=128, conv2d_2_in_channels=128)
        self.module0_3 = Module0(batchnorm2d_0_num_features=160, conv2d_2_in_channels=160)
        self.module0_4 = Module0(batchnorm2d_0_num_features=192, conv2d_2_in_channels=192)
        self.module0_5 = Module0(batchnorm2d_0_num_features=224, conv2d_2_in_channels=224)
        self.module0_6 = Module0(batchnorm2d_0_num_features=256, conv2d_2_in_channels=256)
        self.module0_7 = Module0(batchnorm2d_0_num_features=288, conv2d_2_in_channels=288)
        self.module0_8 = Module0(batchnorm2d_0_num_features=320, conv2d_2_in_channels=320)
        self.module0_9 = Module0(batchnorm2d_0_num_features=352, conv2d_2_in_channels=352)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        opt_batchnorm2d_0 = self.batchnorm2d_0(module0_1_opt)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_avgpool2d_3 = self.pad_avgpool2d_3(opt_conv2d_2)
        opt_avgpool2d_3 = self.avgpool2d_3(opt_avgpool2d_3)
        module0_2_opt = self.module0_2(opt_avgpool2d_3)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        module0_8_opt = self.module0_8(module0_7_opt)
        module0_9_opt = self.module0_9(module0_8_opt)
        return module0_9_opt


class Module48(nn.Cell):

    def __init__(
            self, batchnorm2d_0_num_features, conv2d_2_in_channels, conv2d_2_out_channels,
            module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels, module0_1_batchnorm2d_0_num_features,
            module0_1_conv2d_2_in_channels, module0_2_batchnorm2d_0_num_features, module0_2_conv2d_2_in_channels,
            module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels, module0_4_batchnorm2d_0_num_features,
            module0_4_conv2d_2_in_channels, module0_5_batchnorm2d_0_num_features, module0_5_conv2d_2_in_channels,
            module0_6_batchnorm2d_0_num_features, module0_6_conv2d_2_in_channels, module0_7_batchnorm2d_0_num_features,
            module0_7_conv2d_2_in_channels, module0_8_batchnorm2d_0_num_features, module0_8_conv2d_2_in_channels,
            module0_9_batchnorm2d_0_num_features, module0_9_conv2d_2_in_channels, module0_10_batchnorm2d_0_num_features,
            module0_10_conv2d_2_in_channels, module0_11_batchnorm2d_0_num_features, module0_11_conv2d_2_in_channels,
            module0_12_batchnorm2d_0_num_features, module0_12_conv2d_2_in_channels,
            module0_13_batchnorm2d_0_num_features, module0_13_conv2d_2_in_channels,
            module0_14_batchnorm2d_0_num_features, module0_14_conv2d_2_in_channels,
            module0_15_batchnorm2d_0_num_features, module0_15_conv2d_2_in_channels,
            module0_16_batchnorm2d_0_num_features, module0_16_conv2d_2_in_channels,
            module0_17_batchnorm2d_0_num_features, module0_17_conv2d_2_in_channels,
            module0_18_batchnorm2d_0_num_features, module0_18_conv2d_2_in_channels,
            module0_19_batchnorm2d_0_num_features, module0_19_conv2d_2_in_channels):
        super(Module48, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.pad_avgpool2d_3 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
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
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        opt_batchnorm2d_0 = self.batchnorm2d_0(module0_3_opt)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_avgpool2d_3 = self.pad_avgpool2d_3(opt_conv2d_2)
        opt_avgpool2d_3 = self.avgpool2d_3(opt_avgpool2d_3)
        module0_4_opt = self.module0_4(opt_avgpool2d_3)
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


class DenseNet_features(nn.Cell):

    def __init__(self):
        super(DenseNet_features, self).__init__()
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
        self.module19_0 = Module19(module0_0_batchnorm2d_0_num_features=64,
                                   module0_0_conv2d_2_in_channels=64,
                                   module0_1_batchnorm2d_0_num_features=96,
                                   module0_1_conv2d_2_in_channels=96,
                                   module0_2_batchnorm2d_0_num_features=128,
                                   module0_2_conv2d_2_in_channels=128,
                                   module0_3_batchnorm2d_0_num_features=160,
                                   module0_3_conv2d_2_in_channels=160)
        self.module43_0 = Module43()
        self.module48_0 = Module48(batchnorm2d_0_num_features=512,
                                   conv2d_2_in_channels=512,
                                   conv2d_2_out_channels=256,
                                   module0_0_batchnorm2d_0_num_features=384,
                                   module0_0_conv2d_2_in_channels=384,
                                   module0_1_batchnorm2d_0_num_features=416,
                                   module0_1_conv2d_2_in_channels=416,
                                   module0_2_batchnorm2d_0_num_features=448,
                                   module0_2_conv2d_2_in_channels=448,
                                   module0_3_batchnorm2d_0_num_features=480,
                                   module0_3_conv2d_2_in_channels=480,
                                   module0_4_batchnorm2d_0_num_features=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_5_batchnorm2d_0_num_features=288,
                                   module0_5_conv2d_2_in_channels=288,
                                   module0_6_batchnorm2d_0_num_features=320,
                                   module0_6_conv2d_2_in_channels=320,
                                   module0_7_batchnorm2d_0_num_features=352,
                                   module0_7_conv2d_2_in_channels=352,
                                   module0_8_batchnorm2d_0_num_features=384,
                                   module0_8_conv2d_2_in_channels=384,
                                   module0_9_batchnorm2d_0_num_features=416,
                                   module0_9_conv2d_2_in_channels=416,
                                   module0_10_batchnorm2d_0_num_features=448,
                                   module0_10_conv2d_2_in_channels=448,
                                   module0_11_batchnorm2d_0_num_features=480,
                                   module0_11_conv2d_2_in_channels=480,
                                   module0_12_batchnorm2d_0_num_features=512,
                                   module0_12_conv2d_2_in_channels=512,
                                   module0_13_batchnorm2d_0_num_features=544,
                                   module0_13_conv2d_2_in_channels=544,
                                   module0_14_batchnorm2d_0_num_features=576,
                                   module0_14_conv2d_2_in_channels=576,
                                   module0_15_batchnorm2d_0_num_features=608,
                                   module0_15_conv2d_2_in_channels=608,
                                   module0_16_batchnorm2d_0_num_features=640,
                                   module0_16_conv2d_2_in_channels=640,
                                   module0_17_batchnorm2d_0_num_features=672,
                                   module0_17_conv2d_2_in_channels=672,
                                   module0_18_batchnorm2d_0_num_features=704,
                                   module0_18_conv2d_2_in_channels=704,
                                   module0_19_batchnorm2d_0_num_features=736,
                                   module0_19_conv2d_2_in_channels=736)
        self.module19_1 = Module19(module0_0_batchnorm2d_0_num_features=768,
                                   module0_0_conv2d_2_in_channels=768,
                                   module0_1_batchnorm2d_0_num_features=800,
                                   module0_1_conv2d_2_in_channels=800,
                                   module0_2_batchnorm2d_0_num_features=832,
                                   module0_2_conv2d_2_in_channels=832,
                                   module0_3_batchnorm2d_0_num_features=864,
                                   module0_3_conv2d_2_in_channels=864)
        self.module48_1 = Module48(batchnorm2d_0_num_features=1024,
                                   conv2d_2_in_channels=1024,
                                   conv2d_2_out_channels=512,
                                   module0_0_batchnorm2d_0_num_features=896,
                                   module0_0_conv2d_2_in_channels=896,
                                   module0_1_batchnorm2d_0_num_features=928,
                                   module0_1_conv2d_2_in_channels=928,
                                   module0_2_batchnorm2d_0_num_features=960,
                                   module0_2_conv2d_2_in_channels=960,
                                   module0_3_batchnorm2d_0_num_features=992,
                                   module0_3_conv2d_2_in_channels=992,
                                   module0_4_batchnorm2d_0_num_features=512,
                                   module0_4_conv2d_2_in_channels=512,
                                   module0_5_batchnorm2d_0_num_features=544,
                                   module0_5_conv2d_2_in_channels=544,
                                   module0_6_batchnorm2d_0_num_features=576,
                                   module0_6_conv2d_2_in_channels=576,
                                   module0_7_batchnorm2d_0_num_features=608,
                                   module0_7_conv2d_2_in_channels=608,
                                   module0_8_batchnorm2d_0_num_features=640,
                                   module0_8_conv2d_2_in_channels=640,
                                   module0_9_batchnorm2d_0_num_features=672,
                                   module0_9_conv2d_2_in_channels=672,
                                   module0_10_batchnorm2d_0_num_features=704,
                                   module0_10_conv2d_2_in_channels=704,
                                   module0_11_batchnorm2d_0_num_features=736,
                                   module0_11_conv2d_2_in_channels=736,
                                   module0_12_batchnorm2d_0_num_features=768,
                                   module0_12_conv2d_2_in_channels=768,
                                   module0_13_batchnorm2d_0_num_features=800,
                                   module0_13_conv2d_2_in_channels=800,
                                   module0_14_batchnorm2d_0_num_features=832,
                                   module0_14_conv2d_2_in_channels=832,
                                   module0_15_batchnorm2d_0_num_features=864,
                                   module0_15_conv2d_2_in_channels=864,
                                   module0_16_batchnorm2d_0_num_features=896,
                                   module0_16_conv2d_2_in_channels=896,
                                   module0_17_batchnorm2d_0_num_features=928,
                                   module0_17_conv2d_2_in_channels=928,
                                   module0_18_batchnorm2d_0_num_features=960,
                                   module0_18_conv2d_2_in_channels=960,
                                   module0_19_batchnorm2d_0_num_features=992,
                                   module0_19_conv2d_2_in_channels=992)
        self.batchnorm2d_363 = nn.BatchNorm2d(num_features=1024, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_364 = nn.ReLU()
        self.kernel_sizes = [7, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
        self.strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module19_0_opt = self.module19_0(opt_maxpool2d_2)
        module43_0_opt = self.module43_0(module19_0_opt)
        module48_0_opt = self.module48_0(module43_0_opt)
        module19_1_opt = self.module19_1(module48_0_opt)
        module48_1_opt = self.module48_1(module19_1_opt)
        opt_batchnorm2d_363 = self.batchnorm2d_363(module48_1_opt)
        opt_relu_364 = self.relu_364(opt_batchnorm2d_363)
        return opt_relu_364

def densnet121_features():
    densnet121_model = DenseNet_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_densnet121_model_path)
    mindspore.load_param_into_net(densnet121_model,param_dict)
    print("Load densnet121_model model pretrained from ImageNet")
    return densnet121_model

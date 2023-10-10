import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_vgg16_path = "../imagenet_pretrained_weight/vgg16.ckpt"

class Module4(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module4, self).__init__()
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

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
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
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        return opt_relu_5


class Module7(nn.Cell):

    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels):
        super(Module7, self).__init__()
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)

    def construct(self, x):
        opt_maxpool2d_0 = self.pad_maxpool2d_0(x)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        module0_0_opt = self.module0_0(opt_maxpool2d_0)
        return module0_0_opt


class VGG_features(nn.Cell):

    def __init__(self):
        super(VGG_features, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module4_0 = Module4(conv2d_0_in_channels=64, conv2d_0_out_channels=64)
        self.pad_maxpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module4_1 = Module4(conv2d_0_in_channels=64, conv2d_0_out_channels=128)
        self.module4_2 = Module4(conv2d_0_in_channels=128, conv2d_0_out_channels=128)
        self.module7_0 = Module7(module0_0_conv2d_0_in_channels=128,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_2_in_channels=256,
                                 module0_0_conv2d_2_out_channels=256,
                                 module0_0_conv2d_4_in_channels=256,
                                 module0_0_conv2d_4_out_channels=256)
        self.module7_1 = Module7(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=512)
        self.module7_2 = Module7(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=512)
        self.pad_maxpool2d_30 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_30 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.kernel_sizes = [3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2]
        self.strides = [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]
        self.paddings = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module4_0_opt = self.module4_0(opt_relu_1)
        opt_maxpool2d_4 = self.pad_maxpool2d_4(module4_0_opt)
        opt_maxpool2d_4 = self.maxpool2d_4(opt_maxpool2d_4)
        module4_1_opt = self.module4_1(opt_maxpool2d_4)
        module4_2_opt = self.module4_2(module4_1_opt)
        module7_0_opt = self.module7_0(module4_2_opt)
        module7_1_opt = self.module7_1(module7_0_opt)
        module7_2_opt = self.module7_2(module7_1_opt)
        opt_maxpool2d_30 = self.pad_maxpool2d_30(module7_2_opt)
        opt_maxpool2d_30 = self.maxpool2d_30(opt_maxpool2d_30)
        return opt_maxpool2d_30

def vgg16_features():
    vgg16_model = VGG_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_vgg16_path)
    mindspore.load_param_into_net(vgg16_model,param_dict)
    print("Load VGG16 model pretrained from ImageNet")
    return vgg16_model

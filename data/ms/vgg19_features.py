import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

IMAGENET_pretrained_vgg19_path = "../imagenet_pretrained_weight/vgg19.ckpt"

class Module3(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module3, self).__init__()
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


class Module2(nn.Cell):

    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels, conv2d_3_in_channels, conv2d_3_out_channels,
                 conv2d_5_in_channels, conv2d_5_out_channels):
        super(Module2, self).__init__()
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
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

    def construct(self, x):
        opt_maxpool2d_0 = self.pad_maxpool2d_0(x)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_maxpool2d_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        return opt_relu_6


class Module5(nn.Cell):

    def __init__(self, module3_0_conv2d_0_in_channels, module3_0_conv2d_0_out_channels, module2_0_conv2d_1_in_channels,
                 module2_0_conv2d_1_out_channels, module2_0_conv2d_3_in_channels, module2_0_conv2d_3_out_channels,
                 module2_0_conv2d_5_in_channels, module2_0_conv2d_5_out_channels):
        super(Module5, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels)
        self.module2_0 = Module2(conv2d_1_in_channels=module2_0_conv2d_1_in_channels,
                                 conv2d_1_out_channels=module2_0_conv2d_1_out_channels,
                                 conv2d_3_in_channels=module2_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module2_0_conv2d_3_out_channels,
                                 conv2d_5_in_channels=module2_0_conv2d_5_in_channels,
                                 conv2d_5_out_channels=module2_0_conv2d_5_out_channels)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module2_0_opt = self.module2_0(module3_0_opt)
        return module2_0_opt


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
        self.module3_0 = Module3(conv2d_0_in_channels=64, conv2d_0_out_channels=64)
        self.pad_maxpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module3_1 = Module3(conv2d_0_in_channels=64, conv2d_0_out_channels=128)
        self.module5_0 = Module5(module3_0_conv2d_0_in_channels=128,
                                 module3_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_1_in_channels=128,
                                 module2_0_conv2d_1_out_channels=256,
                                 module2_0_conv2d_3_in_channels=256,
                                 module2_0_conv2d_3_out_channels=256,
                                 module2_0_conv2d_5_in_channels=256,
                                 module2_0_conv2d_5_out_channels=256)
        self.module5_1 = Module5(module3_0_conv2d_0_in_channels=256,
                                 module3_0_conv2d_0_out_channels=256,
                                 module2_0_conv2d_1_in_channels=256,
                                 module2_0_conv2d_1_out_channels=512,
                                 module2_0_conv2d_3_in_channels=512,
                                 module2_0_conv2d_3_out_channels=512,
                                 module2_0_conv2d_5_in_channels=512,
                                 module2_0_conv2d_5_out_channels=512)
        self.module5_2 = Module5(module3_0_conv2d_0_in_channels=512,
                                 module3_0_conv2d_0_out_channels=512,
                                 module2_0_conv2d_1_in_channels=512,
                                 module2_0_conv2d_1_out_channels=512,
                                 module2_0_conv2d_3_in_channels=512,
                                 module2_0_conv2d_3_out_channels=512,
                                 module2_0_conv2d_5_in_channels=512,
                                 module2_0_conv2d_5_out_channels=512)
        self.module3_2 = Module3(conv2d_0_in_channels=512, conv2d_0_out_channels=512)
        self.pad_maxpool2d_36 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_36 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.kernel_sizes = [3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2]
        self.strides = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2]
        self.paddings = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module3_0_opt = self.module3_0(opt_relu_1)
        opt_maxpool2d_4 = self.pad_maxpool2d_4(module3_0_opt)
        opt_maxpool2d_4 = self.maxpool2d_4(opt_maxpool2d_4)
        module3_1_opt = self.module3_1(opt_maxpool2d_4)
        module5_0_opt = self.module5_0(module3_1_opt)
        module5_1_opt = self.module5_1(module5_0_opt)
        module5_2_opt = self.module5_2(module5_1_opt)
        module3_2_opt = self.module3_2(module5_2_opt)
        opt_maxpool2d_36 = self.pad_maxpool2d_36(module3_2_opt)
        opt_maxpool2d_36 = self.maxpool2d_36(opt_maxpool2d_36)
        return opt_maxpool2d_36
def vgg19_features():
    vgg19_model = VGG_features()
    param_dict = mindspore.load_checkpoint(IMAGENET_pretrained_vgg19_path)
    mindspore.load_param_into_net(vgg19_model,param_dict)
    print("Load VGG19 model pretrained from ImageNet")
    return vgg19_model

if __name__ == '__main__':
    vgg19  = vgg19_features()
    print(vgg19.conv_info())

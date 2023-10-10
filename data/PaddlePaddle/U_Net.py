import paddle.fluid as fluid
from config import N_CLASSES
import numpy as np


class Conv_block(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(Conv_block, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=1,
                                          padding=1)
        self.conv2 = fluid.dygraph.Conv2D(num_channels=num_filters, num_filters=num_filters, filter_size=3, stride=1,
                                          padding=1)
        self.bn = fluid.dygraph.BatchNorm(num_filters)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = fluid.layers.relu6(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = fluid.layers.relu6(x)
        return x


class U_net(fluid.dygraph.Layer):
    def __init__(self):
        super(U_net, self).__init__()
        self.conv_block1 = Conv_block(num_channels=3, num_filters=64)
        self.conv_block2 = Conv_block(num_channels=64, num_filters=128)
        self.conv_block3 = Conv_block(num_channels=128, num_filters=256)
        self.conv_block4 = Conv_block(num_channels=256, num_filters=512)
        self.conv_block5 = Conv_block(num_channels=512, num_filters=1024)

        self.conv_block4_reverse = Conv_block(num_channels=1024 + 512, num_filters=512)
        self.conv_block3_reverse = Conv_block(num_channels=512 + 256, num_filters=256)
        self.conv_block2_reverse = Conv_block(num_channels=256 + 128, num_filters=128)
        self.conv_block1_reverse = Conv_block(num_channels=128 + 64, num_filters=64)
        self.conv0 = fluid.dygraph.Conv2D(num_channels=64, num_filters=N_CLASSES, filter_size=3, stride=1, padding=1)

    def forward(self, img):
        f1 = self.conv_block1(img)  # 576 * 576 * 3 -> 576 * 576 * 64
        f2 = fluid.layers.pool2d(f1, pool_size=2, pool_stride=2, pool_type="max")  # 288 * 288 * 64
        f2 = self.conv_block2(f2)  # 288 * 288 * 128
        f2 = fluid.layers.dropout(f2, 0.4)

        f3 = fluid.layers.pool2d(f2, pool_size=2, pool_stride=2, pool_type="max")  # 144 * 144 * 128
        f3 = self.conv_block3(f3)  # 144 * 144 * 256
        f3 = fluid.layers.dropout(f3, 0.4)

        f4 = fluid.layers.pool2d(f3, pool_size=2, pool_stride=2, pool_type="max")  # 72 * 72 * 256
        f4 = self.conv_block4(f4)  # 72 * 72 * 512
        f4 = fluid.layers.dropout(f4, 0.4)

        f5 = fluid.layers.pool2d(f4, pool_size=2, pool_stride=2, pool_type="max")  # 36 * 36 * 512
        f5 = self.conv_block5(f5)  # 36 * 36 * 1024
        f5 = fluid.layers.dropout(f5, 0.4)

        f4_reverse = fluid.layers.image_resize(f5, scale=2)  # 72 * 72 * 1024
        f4_reverse = self.conv_block4_reverse(
            fluid.layers.concat([f4_reverse, f4], axis=1))  # 72 * 72 * (512 + 1024) -> 72 * 72 * 512

        f3_reverse = fluid.layers.image_resize(f4_reverse, scale=2)  # 144 * 144 * 512
        f3_reverse = self.conv_block3_reverse(
            fluid.layers.concat([f3_reverse, f3], axis=1))  # 144 * 144 * (512 + 256) -> 144 * 144 * 256

        f2_reverse = fluid.layers.image_resize(f3_reverse, scale=2)  # 288 * 288 * 256
        f2_reverse = self.conv_block2_reverse(
            fluid.layers.concat([f2_reverse, f2], axis=1))  # 288 * 288 * (256 + 128) -> 288 * 288 * 128

        f1_reverse = fluid.layers.image_resize(f2_reverse, scale=2)  # 576 * 576 * 128
        f1_reverse = self.conv_block1_reverse(
            fluid.layers.concat([f1_reverse, f1], axis=1))  # 576 * 576 * (64 + 128) -> 576 * 576 * 64

        f0 = self.conv0(f1_reverse)
        f0 = fluid.layers.softmax(f0, axis=1)
        return f0


if __name__ == "__main__":
    with fluid.dygraph.guard():
        img = np.zeros([1, 3, 512, 512]).astype('float32')
        model = U_net()
        img = fluid.dygraph.to_variable(img)
        outs = model(img)
        outs = fluid.layers.transpose(outs, perm=(0, 2, 3, 1))
        print(outs.numpy()[0][0][0])
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Layer


class Encoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()

        self.conv1 = Conv2D(num_channels,
                            num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = BatchNorm(num_filters, act='relu')

        self.conv2 = Conv2D(num_filters,
                            num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn2 = BatchNorm(num_filters, act='relu')

        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type='max', ceil_mode=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_pooled = self.pool(x)

        return x, x_pooled


class Decoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Decoder, self).__init__()
        self.up = Conv2DTranspose(num_channels=num_channels,
                                  num_filters=num_filters,
                                  filter_size=2,
                                  stride=2)

        self.conv1 = Conv2D(num_channels,
                            num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = BatchNorm(num_filters, act='relu')

        self.conv2 = Conv2D(num_filters,
                            num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn2 = BatchNorm(num_filters, act='relu')

    def forward(self, inputs_prev, inputs):
        x = self.up(inputs)
        h_diff = (inputs_prev.shape[2] - x.shape[2])
        w_diff = (inputs_prev.shape[3] - x.shape[3])
        x = fluid.layers.pad2d(x, paddings=[h_diff // 2, h_diff - h_diff // 2, w_diff // 2, w_diff - w_diff // 2])
        x = fluid.layers.concat([inputs_prev, x], axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UNet(Layer):
    def __init__(self, num_classes=59):
        super(UNet, self).__init__()

        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)

        self.mid_conv1 = Conv2D(512, 1024, filter_size=1, padding=0, stride=1)
        self.mid_bn1 = BatchNorm(1024, act='relu')
        self.mid_conv2 = Conv2D(1024, 1024, filter_size=1, padding=0, stride=1)
        self.mid_bn2 = BatchNorm(1024, act='relu')

        self.up4 = Decoder(1024, 512)
        self.up3 = Decoder(512, 256)
        self.up2 = Decoder(256, 128)
        self.up1 = Decoder(128, 64)

        self.last_conv = Conv2D(num_channels=64, num_filters=num_classes, filter_size=1)

    def forward(self, inputs):
        x1, x = self.down1(inputs)
        print(x1.shape, x.shape)
        x2, x = self.down2(x)
        print(x2.shape, x.shape)
        x3, x = self.down3(x)
        print(x3.shape, x.shape)
        x4, x = self.down4(x)
        print(x4.shape, x.shape)

        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        print(x4.shape, x.shape)
        x = self.up4(x4, x)
        print(x3.shape, x.shape)
        x = self.up3(x3, x)
        print(x2.shape, x.shape)
        x = self.up2(x2, x)
        print(x1.shape, x.shape)
        x = self.up1(x1, x)
        print(x.shape)

        x = self.last_conv(x)

        print(x.shape)
        return x


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        model = UNet(num_classes=5)
        x_data = np.random.rand(1, 3, 123, 123).astype(np.float32)
        inputs = to_variable(x_data)
        pred = model(inputs)

        print(pred.shape)


if __name__ == '__main__':
    main()
import paddle
import paddle.nn as nn


class RC_block(nn.Layer):
    def __init__(self, channel, t=2):
        super().__init__()
        self.t = t

        self.conv = nn.Sequential(
            nn.Conv2D(channel, channel, 3, 1, 1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

    def forward(self, x):
        r_x = self.conv(x)

        for _ in range(self.t):
            r_x = self.conv(x + r_x)

        return r_x


class RRC_block(nn.Layer):
    def __init__(self, channel, t=2):
        super().__init__()

        self.RC_net = nn.Sequential(
            RC_block(channel, t=t),
            RC_block(channel, t=t),
        )

    def forward(self, x):
        res_x = self.RC_net(x)

        return x + res_x


class R2UNet(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            RRC_block(64),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            RRC_block(128),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(128, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            RRC_block(256),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            RRC_block(512),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(512, 1024, 3, 1, 1),
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            RRC_block(1024),
            nn.Conv2DTranspose(1024, 512, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2D(1024, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            RRC_block(512),
            nn.Conv2DTranspose(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2D(512, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            RRC_block(256),
            nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2D(256, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            RRC_block(128),
            nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2D(128, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            RRC_block(64),
            nn.Conv2D(64, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)

        x = self.up_conv1(paddle.concat((x, x4), axis=1))
        x = self.up_conv2(paddle.concat((x, x3), axis=1))
        x = self.up_conv3(paddle.concat((x, x2), axis=1))
        x = self.final_conv(paddle.concat((x, x1), axis=1))

        x = self.sigmoid(x)

        return x


class UNet(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(128, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(512, 1024, 3, 1, 1),
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.Conv2D(1024, 1024, 3, 1, 1),
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.Conv2D(1024, 1024, 3, 1, 1),
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.Conv2DTranspose(1024, 512, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2D(1024, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2DTranspose(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2D(512, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2D(256, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2D(128, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)

        x = self.up_conv1(paddle.concat((x, x4), axis=1))
        x = self.up_conv2(paddle.concat((x, x3), axis=1))
        x = self.up_conv3(paddle.concat((x, x2), axis=1))
        x = self.final_conv(paddle.concat((x, x1), axis=1))

        x = self.sigmoid(x)

        return x


class MainUNet(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(32, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(128, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2DTranspose(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2D(512, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2D(256, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2D(128, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2DTranspose(64, 32, kernel_size=2, stride=2),
        )

        self.up_conv4 = nn.Sequential(
            nn.Conv2D(64, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            # nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2),
        )

        self.conv_1x1 = nn.Conv2D(32, 1, 1)

    def forward(self, x):
        latent1 = self.conv1(x)
        x2 = self.conv2(latent1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.trans_conv(x4)

        x = self.up_conv1(paddle.concat((x, x4), axis=1))
        x = self.up_conv2(paddle.concat((x, x3), axis=1))
        x = self.up_conv3(paddle.concat((x, x2), axis=1))
        latent2 = self.up_conv4(paddle.concat((x, latent1), axis=1))

        x = self.conv_1x1(latent2)

        return latent1, latent2, paddle.nn.functional.sigmoid(x)


class MiniUNet(nn.Layer):
    def __init__(self, iter=2):
        super().__init__()

        self.transpose = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
        )

        self.dim_reduc = nn.LayerList([nn.Conv2D(64 + 32 * i, 32, 1, 1) for i in range(iter)])

        self.conv1 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(32, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2D(2, stride=2),
            nn.Conv2D(128, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 256, 3, 1, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2D(256, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 64, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2D(128, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2DTranspose(64, 32, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2D(64, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),

        )

        self.conv_1x1 = nn.Conv2D(32, 1, 1)

    def forward(self, latent1, latent2):
        latent3 = self.transpose(latent2)
        latent1 = paddle.concat((latent1, latent3), axis=1)
        idx = int((latent1.size(1) - 64) / 32)
        x1 = self.dim_reduc[idx](latent1)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x = self.trans_conv(x3)

        x = self.up_conv1(paddle.concat((x, x3), axis=1))
        x = self.up_conv2(paddle.concat((x, x2), axis=1))
        latent2 = self.up_conv3(paddle.concat((x, x1), axis=1))

        x = self.conv_1x1(latent2)

        return latent1, latent2, paddle.nn.functional.sigmoid(x)


class IterNet(nn.Layer):
    def __init__(self, t=2):
        super().__init__()
        self.iter = t

        self.main = MainUNet()
        self.mini = MiniUNet()

    def forward(self, x):
        out_list = []
        latent1, latent2, out = self.main(x)
        out_list.append(out)

        for _ in range(self.iter):
            latent1, latent2, out = self.mini(latent1, latent2)
            out_list.append(out)

        return out_list

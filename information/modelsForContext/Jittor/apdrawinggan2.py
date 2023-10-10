import jittor as jt
from jittor import init
from jittor import nn

import pdb


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
        if (hasattr(m, 'bias') and (m.bias is not None)):
            jt.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class ResidualBlock(nn.Module):

    def __init__(self, in_features, dropout=0.5):
        super(ResidualBlock, self).__init__()
        model = [nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3, bias=False), nn.BatchNorm2d(in_features),
                 nn.ReLU()]
        if dropout:
            model += [nn.Dropout(dropout)]
        model += [nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3, bias=False), nn.BatchNorm2d(in_features)]
        self.conv_block = nn.Sequential(*model)

    def execute(self, x):
        return (x + self.conv_block(x))


class GeneratorResNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_res_blocks=9):
        super(GeneratorResNet, self).__init__()
        out_features = 64
        model = [nn.ReflectionPad2d(3), nn.Conv(in_channels, out_features, 7, bias=False), nn.BatchNorm2d(out_features),
                 nn.ReLU()]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [nn.Conv(in_features, out_features, 3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
        for _ in range(num_res_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [nn.ConvTranspose(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv(out_features, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


class GeneratorResStyle2Net(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_res_blocks=9, extra_channel=3):
        super(GeneratorResStyle2Net, self).__init__()
        out_features = 64
        model0 = [nn.ReflectionPad2d(3), nn.Conv(in_channels, out_features, 7, bias=False),
                  nn.BatchNorm2d(out_features), nn.ReLU()]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model0 += [nn.Conv(in_features, out_features, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
        model = [nn.Conv2d(out_features + extra_channel, out_features, 3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(out_features), nn.ReLU()]
        for _ in range(num_res_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [nn.ConvTranspose(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv(out_features, out_channels, 7), nn.Tanh()]
        self.model0 = nn.Sequential(*model0)
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, input1, input2):  # input2 [bs,c]
        f1 = self.model0(input1)
        [bs, c, h, w] = f1.shape
        input2 = input2.repeat(h, w, 1, 1).permute([2, 3, 0, 1])
        y1 = jt.contrib.concat((f1, input2), 1)
        return self.model(y1)


class AutoEncoderWithFC(nn.Module):
    def __init__(self, input_nc, output_nc, h=96, w=96):
        super(AutoEncoderWithFC, self).__init__()

        out_features = 64
        model = [nn.Conv(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False)]
        in_features = out_features
        for _ in range(3):
            out_features *= 2
            model += [nn.LeakyReLU(0.2),
                      nn.Conv(in_features, out_features, 4,
                              stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_features)]
            in_features = out_features
        self.encoder = nn.Sequential(*model)

        self.rh = int(h / 16)
        self.rw = int(w / 16)
        self.feat_dim = 512 * self.rh * self.rw

        self.fc1 = nn.Linear(self.feat_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, self.feat_dim)

        model2 = []
        for _ in range(3):
            out_features //= 2
            model2 += [nn.ReLU(),
                       nn.ConvTranspose(in_features, out_features, 4,
                                        stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(out_features)]
            in_features = out_features
        model2 += [nn.ReLU(),
                   nn.ConvTranspose(out_features, output_nc, 4, stride=2, padding=1, bias=False),
                   nn.Tanh()]
        self.decoder = nn.Sequential(*model2)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        ax = self.encoder(x)
        ax = jt.reshape(ax, [ax.shape[0], self.feat_dim])
        f1 = self.fc1(ax)
        f1 = self.relu(f1)
        f2 = self.fc2(f1)
        f2 = jt.reshape(f2, [f2.shape[0], 512, self.rh, self.rw])
        y = self.decoder(f2)
        return y


class Classifier(nn.Module):
    def __init__(self, input_nc, classes, ngf=64, num_downs=3, h=96, w=96):
        super(Classifier, self).__init__()

        model = [nn.Conv(input_nc, ngf, 4, stride=2, padding=1, bias=False)]
        multiple = 2
        for i in range(num_downs):
            mult = multiple ** i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv(int(ngf * mult), int(ngf * mult * multiple), 4,
                              stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(int(ngf * mult * multiple))]
        self.encoder = nn.Sequential(*model)
        strides = 2 ** (num_downs + 1)
        self.fc1 = nn.Linear(int(ngf * h * w / (strides * 2)), classes)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        ax = self.encoder(x)  # b, 512, 6, 6
        ax = ax.view(ax.size(0), -1)  # view -- reshape
        return self.fc1(ax)


class Combiner(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Combiner, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv(in_channels, 64, 7, padding=0, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU()]

        for i in range(2):
            model += [ResidualBlock(64, dropout=0.5)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv(64, out_channels, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.model = nn.Sequential(*discriminator_block((in_channels + out_channels), 64, normalization=False),
                                   *discriminator_block(64, 128), *discriminator_block(128, 256),
                                   *discriminator_block(256, 512, stride=1), nn.Conv(512, 1, 4, stride=1, padding=1),
                                   nn.Sigmoid())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img_A, img_B):
        img_input = jt.contrib.concat((img_A, img_B), dim=1)
        return self.model(img_input)


class UnetBlock(nn.Module):

    def __init__(self, in_size, out_size, inner_nc, dropout=0.0, innermost=False, outermost=False, submodule=None):
        super(UnetBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv(in_size, inner_nc, 4, stride=2, padding=1, bias=False)
        downnorm = nn.BatchNorm2d(inner_nc)
        downrelu = nn.LeakyReLU(0.2)
        upnorm = nn.BatchNorm2d(out_size)
        uprelu = nn.ReLU()

        if outermost:
            upconv = nn.ConvTranspose(2 * inner_nc, out_size, 4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose(inner_nc, out_size, 4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose(2 * inner_nc, out_size, 4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return jt.contrib.concat((x, self.model(x)), dim=1)


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_downs=8):
        super(GeneratorUNet, self).__init__()

        unet_block = UnetBlock(512, 512, inner_nc=512, submodule=None, innermost=True)  # down8, up1
        for i in range(num_downs - 5):
            unet_block = UnetBlock(512, 512, inner_nc=512, submodule=unet_block, dropout=0.5)
        unet_block = UnetBlock(256, 256, inner_nc=512, submodule=unet_block)  # down4, up5
        unet_block = UnetBlock(128, 128, inner_nc=256, submodule=unet_block)  # down3, up6
        unet_block = UnetBlock(64, 64, inner_nc=128, submodule=unet_block)  # down2, up7
        unet_block = UnetBlock(in_channels, out_channels, inner_nc=64, submodule=unet_block,
                               outermost=True)  # down1, final

        self.model = unet_block

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


class Regressor4(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(Regressor4, self).__init__()

        use_bias = True
        sequence = [
            nn.Conv(input_nc, ngf, 3, stride=1, padding=1, bias=use_bias),  # 11->11
            nn.LeakyReLU(0.2),
            nn.Conv(ngf, ngf * 2, 3, stride=1, padding=1, bias=use_bias),  # 11->11
            nn.LeakyReLU(0.2),
            nn.Conv(ngf * 2, ngf * 4, 3, stride=1, padding=1, bias=use_bias),  # 11->11
            nn.LeakyReLU(0.2),
            nn.Conv(ngf * 4, 1, 11, stride=1, padding=0, bias=use_bias),  # 11->1
        ]

        self.model = nn.Sequential(*sequence)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)
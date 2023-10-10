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

    def __init__(self, in_features, dropout=0.0):
        super(ResidualBlock, self).__init__()
        model = [nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3), nn.BatchNorm2d(in_features), nn.ReLU()]
        if dropout:
            model += [nn.Dropout(dropout)]
        model += [nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3), nn.BatchNorm2d(in_features)]
        self.conv_block = nn.Sequential(*model)

    def execute(self, x):
        return (x + self.conv_block(x))


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(scale=0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose(in_size, out_size, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_size),
                  nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def execute(self, x, skip_input):
        x = self.model(x)
        x = jt.contrib.concat((x, skip_input), dim=1)
        return x


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


class PartUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PartUnet, self).__init__()

        unet_block = UnetBlock(128, 128, inner_nc=256, submodule=None, innermost=True)
        unet_block = UnetBlock(64, 64, inner_nc=128, submodule=unet_block)
        unet_block = UnetBlock(in_channels, out_channels, inner_nc=64, submodule=unet_block, outermost=True)
        self.model = unet_block

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


class PartUnet2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PartUnet2, self).__init__()

        unet_block = UnetBlock(128, 128, inner_nc=128, submodule=None, innermost=True)
        unet_block = UnetBlock(128, 128, inner_nc=128, submodule=unet_block)
        unet_block = UnetBlock(64, 64, inner_nc=128, submodule=unet_block)
        unet_block = UnetBlock(in_channels, out_channels, inner_nc=64, submodule=unet_block, outermost=True)
        self.model = unet_block

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


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
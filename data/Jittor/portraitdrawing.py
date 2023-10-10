import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
from collections import namedtuple
import utils


def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
        if (hasattr(m, 'bias') and (m.bias is not None)):
            init.constant_(m.bias, value=0.0)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3),
                                        nn.InstanceNorm2d(in_features, affine=None), nn.ReLU(), nn.ReflectionPad2d(1),
                                        nn.Conv(in_features, in_features, 3),
                                        nn.InstanceNorm2d(in_features, affine=None))

    def execute(self, x):
        return (x + self.conv_block(x))


class GeneratorResNet(nn.Module):

    def __init__(self, input_shape, output_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        in_channels = input_shape[0]
        out_channels = output_shape[0]
        out_features = 64
        model = [nn.ReflectionPad2d(3), nn.Conv(in_channels, out_features, 7),
                 nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [nn.Conv(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [nn.ConvTranspose(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv(out_features, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)


class GeneratorResStyleNet(nn.Module):

    def __init__(self, input_shape, output_shape, num_residual_blocks, extra_channel):
        super(GeneratorResStyleNet, self).__init__()
        in_channels = input_shape[0]
        out_channels = output_shape[0]
        out_features = 64
        model0 = [nn.ReflectionPad2d(3), nn.Conv(in_channels, out_features, 7),
                  nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model0 += [nn.Conv(in_features, out_features, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        model = [nn.Conv(out_features + extra_channel, out_features, 3, stride=1, padding=1),
                 nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [nn.ConvTranspose(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv(out_features, out_channels, 7), nn.Tanh()]
        self.model0 = nn.Sequential(*model0)
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x, f):
        f1 = self.model0(x)
        y1 = jt.contrib.concat((f1, f), dim=1)
        return self.model(y1)


class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        (channels, height, width) = input_shape
        self.output_shape = (1, (height // (2 ** 3) - 2), (width // (2 ** 3) - 2))

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=None))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.model = nn.Sequential(*discriminator_block(channels, 64, normalize=False), *discriminator_block(64, 128),
                                   *discriminator_block(128, 256), *discriminator_block(256, 512, stride=1),
                                   nn.Conv(512, 1, 4, padding=1))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)


class DiscriminatorCls(nn.Module):

    def __init__(self, input_shape, n_class):
        super(DiscriminatorCls, self).__init__()
        (channels, height, width) = input_shape
        self.output_shape = (1, (height // (2 ** 3) - 2), (width // (2 ** 3) - 2))

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=None))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.model0 = nn.Sequential(*discriminator_block(channels, 64, normalize=False), *discriminator_block(64, 128),
                                    *discriminator_block(128, 256))
        self.model1 = nn.Sequential(*discriminator_block(256, 512, stride=1), nn.Conv(512, 1, 4, padding=1))
        self.model2 = nn.Sequential(*discriminator_block(256, 512, stride=2), *discriminator_block(512, 512, stride=2),
                                    nn.Conv(512, n_class, 16, padding=0))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        feat = self.model0(img)
        patch = self.model1(feat)
        classl = self.model2(feat)
        return patch, classl.view(classl.size(0), -1)


class HED(nn.Module):

    def __init__(self):
        super(HED, self).__init__()
        self.moduleVggOne = nn.Sequential(nn.Conv(3, 64, 3, stride=1, padding=1), nn.ReLU(),
                                          nn.Conv(64, 64, 3, stride=1, padding=1), nn.ReLU())
        self.moduleVggTwo = nn.Sequential(nn.Pool(2, stride=2, op='maximum'), nn.Conv(64, 128, 3, stride=1, padding=1),
                                          nn.ReLU(), nn.Conv(128, 128, 3, stride=1, padding=1), nn.ReLU())
        self.moduleVggThr = nn.Sequential(nn.Pool(2, stride=2, op='maximum'), nn.Conv(128, 256, 3, stride=1, padding=1),
                                          nn.ReLU(), nn.Conv(256, 256, 3, stride=1, padding=1), nn.ReLU(),
                                          nn.Conv(256, 256, 3, stride=1, padding=1), nn.ReLU())
        self.moduleVggFou = nn.Sequential(nn.Pool(2, stride=2, op='maximum'), nn.Conv(256, 512, 3, stride=1, padding=1),
                                          nn.ReLU(), nn.Conv(512, 512, 3, stride=1, padding=1), nn.ReLU(),
                                          nn.Conv(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.moduleVggFiv = nn.Sequential(nn.Pool(2, stride=2, op='maximum'), nn.Conv(512, 512, 3, stride=1, padding=1),
                                          nn.ReLU(), nn.Conv(512, 512, 3, stride=1, padding=1), nn.ReLU(),
                                          nn.Conv(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.moduleScoreOne = nn.Conv(64, 1, 1, stride=1, padding=0)
        self.moduleScoreTwo = nn.Conv(128, 1, 1, stride=1, padding=0)
        self.moduleScoreThr = nn.Conv(256, 1, 1, stride=1, padding=0)
        self.moduleScoreFou = nn.Conv(512, 1, 1, stride=1, padding=0)
        self.moduleScoreFiv = nn.Conv(512, 1, 1, stride=1, padding=0)
        self.moduleCombine = nn.Sequential(nn.Conv(5, 1, 1, stride=1, padding=0), nn.Sigmoid())

    def execute(self, tensorInput):
        tensorBlue = ((tensorInput[:, 2:3, :, :] * 255.0) - 104.00698793)
        tensorGreen = ((tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762)
        tensorRed = ((tensorInput[:, 0:1, :, :] * 255.0) - 122.67891434)
        tensorInput = jt.contrib.concat([tensorBlue, tensorGreen, tensorRed], dim=1)
        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)
        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)
        tensorScoreOne = nn.Resize((tensorInput.shape[2], tensorInput.shape[3]), mode='bilinear', align_corners=False)(
            tensorScoreOne)
        tensorScoreTwo = nn.Resize((tensorInput.shape[2], tensorInput.shape[3]), mode='bilinear', align_corners=False)(
            tensorScoreTwo)
        tensorScoreThr = nn.Resize((tensorInput.shape[2], tensorInput.shape[3]), mode='bilinear', align_corners=False)(
            tensorScoreThr)
        tensorScoreFou = nn.Resize((tensorInput.shape[2], tensorInput.shape[3]), mode='bilinear', align_corners=False)(
            tensorScoreFou)
        tensorScoreFiv = nn.Resize((tensorInput.shape[2], tensorInput.shape[3]), mode='bilinear', align_corners=False)(
            tensorScoreFiv)
        return self.moduleCombine(
            jt.contrib.concat([tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv], dim=1))


class alexnet(nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if (not requires_grad):
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class NetLinLayer(nn.Module):

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = ([nn.Dropout()] if use_dropout else [])
        layers += [nn.Conv(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = jt.float32([(- 0.03), (- 0.088), (- 0.188)])[None, :, None, None]
        self.scale = jt.float32([0.458, 0.448, 0.45])[None, :, None, None]

    def execute(self, inp):
        return ((inp - self.shift) / self.scale)


def spatial_average(in_tens, keepdims=True):
    return in_tens.mean([2, 3], keepdims=keepdims)


class PNetLin(nn.Module):

    def __init__(self):
        super(PNetLin, self).__init__()
        self.net = alexnet(pretrained=True, requires_grad=False)
        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.scaling_layer = ScalingLayer()

    def execute(self, in0, in1):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.execute(in0_input), self.net.execute(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = utils.normalize_tensor(outs0[kk]), utils.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdims=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        return val
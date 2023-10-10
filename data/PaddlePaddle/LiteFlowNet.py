# ------------------------------------------------------------------------
# Modified from https://github.com/sniklaus/pytorch-liteflownet
# ------------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg
import paddle.fluid.contrib.layers.nn as nn


def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return dg.Conv2D(num_channels=in_channels,
                     num_filters=out_channels,
                     filter_size=kernel_size,
                     stride=stride,
                     padding=padding)


def deconv(in_channels, out_channels, groups):
    return dg.Conv2DTranspose(num_channels=in_channels,
                              num_filters=out_channels,
                              filter_size=4,
                              padding=1,
                              stride=2,
                              groups=groups,
                              bias_attr=False)


def conv1x1(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 1, stride, 0)


def conv3x3(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 3, stride, 1)


class LeakyReLU(dg.Layer):

    def __init__(self, negative_slope=0.1):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return L.leaky_relu(x, alpha=self.negative_slope)


def grid_sample(x, grid):
    h, w = x.shape[2:]
    grid_x = grid[:, :, :, 0:1] * (w / (w - 1))
    grid_y = grid[:, :, :, 1:2] * (h / (h - 1))
    grid = L.concat([grid_x, grid_y], 3)
    return L.grid_sampler(x, grid)


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    h, w = tenFlow.shape[2:]
    if str(tenFlow.shape) not in backwarp_tenGrid:
        L.linspace(0, 1, 128, dtype="float32")
        tenHor = L.linspace(-1.0 + (1.0 / w), 1.0 - (1.0 / w),
                            w, dtype="float32")  # (w, )
        tenVer = L.linspace(-1.0 + (1.0 / h), 1.0 - (1.0 / h),
                            h, dtype="float32")  # (h, )
        tenHor = L.reshape(tenHor, (1, 1, 1, -1))  # (1, 1, 1, w)
        tenHor = L.expand(tenHor, (1, 1, h, 1))  # (1, 1, h, w)
        tenVer = L.reshape(tenVer, (1, 1, -1, 1))  # (1, 1, h, 1)
        tenVer = L.expand(tenVer, (1, 1, 1, w))  # (1, 1, h, w)

        backwarp_tenGrid[str(tenFlow.shape)] = L.concat([tenHor, tenVer], 1)  # (1, 2, h, w)

    tenFlow = L.concat([tenFlow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                        tenFlow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
                        ], 1)  # (1, 2, h, w)
    grid = backwarp_tenGrid[str(tenFlow.shape)] + tenFlow  # (1, 2, h, w)
    grid = L.transpose(grid, (0, 2, 3, 1))
    return grid_sample(tenInput, grid, )


class Features(dg.Layer):

    def __init__(self):
        super(Features, self).__init__()

        self.moduleOne = dg.Sequential(
            ('0', conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)),
            ('1', LeakyReLU())
        )

        self.moduleTwo = dg.Sequential(
            ('0', conv3x3(in_channels=32, out_channels=32, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=32, out_channels=32)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=32, out_channels=32)),
            ('5', LeakyReLU())
        )

        self.moduleThr = dg.Sequential(
            ('0', conv3x3(in_channels=32, out_channels=64, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=64, out_channels=64)),
            ('3', LeakyReLU())
        )

        self.moduleFou = dg.Sequential(
            ('0', conv3x3(in_channels=64, out_channels=96, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=96, out_channels=96)),
            ('3', LeakyReLU())
        )

        self.moduleFiv = dg.Sequential(
            ('0', conv3x3(in_channels=96, out_channels=128, stride=2)),
            ('1', LeakyReLU())
        )

        self.moduleSix = dg.Sequential(
            ('0', conv3x3(in_channels=128, out_channels=192, stride=2)),
            ('1', LeakyReLU())
        )

    def forward(self, tenInput):
        tenOne = self.moduleOne(tenInput)
        tenTwo = self.moduleTwo(tenOne)
        tenThr = self.moduleThr(tenTwo)
        tenFou = self.moduleFou(tenThr)
        tenFiv = self.moduleFiv(tenFou)
        tenSix = self.moduleSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


class Matching(dg.Layer):

    def __init__(self, intLevel):
        super(Matching, self).__init__()

        self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        if intLevel != 2:
            self.moduleFeat = dg.Sequential()
        elif intLevel == 2:
            self.moduleFeat = dg.Sequential(
                ('0', conv1x1(in_channels=32, out_channels=64)),
                ('1', LeakyReLU())
            )

        if intLevel == 6:
            self.moduleUpflow = None
        elif intLevel != 6:
            self.moduleUpflow = deconv(in_channels=2, out_channels=2, groups=2)

        if intLevel >= 4:
            self.moduleUpcorr = None
        elif intLevel < 4:
            self.moduleUpcorr = deconv(in_channels=49, out_channels=49, groups=49)

        self.moduleMain = dg.Sequential(
            ('0', conv3x3(in_channels=49, out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=64)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=64, out_channels=32)),
            ('5', LeakyReLU()),
            ('6', conv2d(in_channels=32, out_channels=2,
                         kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                         padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
        )

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)
        tenFeaturesSecond = self.moduleFeat(tenFeaturesSecond)

        if tenFlow is not None:
            tenFlow = self.moduleUpflow(tenFlow)

        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond,
                                         tenFlow=tenFlow * self.fltBackwarp)

        if self.moduleUpcorr is None:
            correlation = nn.correlation(tenFeaturesFirst, tenFeaturesSecond,
                                         pad_size=3,
                                         kernel_size=1,
                                         max_displacement=3,
                                         stride1=1,
                                         stride2=1, )
            tenCorrelation = L.leaky_relu(correlation, alpha=0.1)
        elif self.moduleUpcorr is not None:
            correlation = nn.correlation(tenFeaturesFirst, tenFeaturesSecond,
                                         pad_size=6,
                                         kernel_size=1,
                                         max_displacement=6,
                                         stride1=2,
                                         stride2=2, )
            tenCorrelation = L.leaky_relu(correlation, alpha=0.1)
            tenCorrelation = self.moduleUpcorr(tenCorrelation)

        return (tenFlow if tenFlow is not None else 0.0) + self.moduleMain(tenCorrelation)


class Subpixel(dg.Layer):

    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        if intLevel != 2:
            self.moduleFeat = dg.Sequential()
        elif intLevel == 2:
            self.moduleFeat = dg.Sequential(
                ('0', conv1x1(in_channels=32, out_channels=64)),
                ('1', LeakyReLU())
            )

        self.moduleMain = dg.Sequential(
            ('0', conv3x3(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=64)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=64, out_channels=32)),
            ('5', LeakyReLU()),
            ('6', conv2d(in_channels=32, out_channels=2,
                         kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                         padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
        )

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)
        tenFeaturesSecond = self.moduleFeat(tenFeaturesSecond)

        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackward)

        tenFeatures = L.concat([tenFeaturesFirst, tenFeaturesSecond, tenFlow], 1)
        return (tenFlow if tenFlow is not None else 0.0) + self.moduleMain(tenFeatures)


class Regularization(dg.Layer):

    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
        self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

        if intLevel >= 5:
            self.moduleFeat = dg.Sequential()
        elif intLevel < 5:
            self.moduleFeat = dg.Sequential(
                ('0', conv1x1(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128)),
                ('1', LeakyReLU())
            )

        self.moduleMain = dg.Sequential(
            ('0', conv3x3(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=128)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=128, out_channels=64)),
            ('5', LeakyReLU()),
            ('6', conv3x3(in_channels=64, out_channels=64)),
            ('7', LeakyReLU()),
            ('8', conv3x3(in_channels=64, out_channels=32)),
            ('9', LeakyReLU()),
            ('10', conv3x3(in_channels=32, out_channels=32)),
            ('11', LeakyReLU())
        )

        if intLevel >= 5:
            self.moduleDist = dg.Sequential(
                ('0', conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                             kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                             padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
            )
        elif intLevel < 5:
            self.moduleDist = dg.Sequential(
                ('0', conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                             kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1,
                             padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0))),
                ('1', conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                             out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                             kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]),
                             stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])))
            )

        self.moduleScaleX = conv1x1(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1)
        self.moduleScaleY = conv1x1(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1)

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        b, _, h, w = tenFlow.shape
        tenDifference = tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackward)
        tenDifference = L.pow(tenDifference, 2)
        tenDifference = L.reduce_sum(tenDifference, 1, True)  # [b, 1, h, w]
        tenDifference = L.sqrt(tenDifference).detach()

        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)

        tenMean = L.reshape(tenFlow, (b, 2, -1))  # [b, 2, h * w]
        tenMean = L.reduce_mean(tenMean, 2, True)  # [b, 2, 1]
        tenMean = L.reshape(tenMean, (b, 2, 1, 1))  # [b, 2, 1, 1]
        tenMean = L.expand(tenMean, (1, 1, h, w))  # [b, 2, h, w]
        delta = tenFlow - tenMean

        diff = L.concat([tenDifference, delta, tenFeaturesFirst], 1)
        tenDist = self.moduleDist(self.moduleMain(diff))
        tenDist = L.pow(tenDist, 2.0) * -1.0
        tenDist = tenDist - L.reduce_max(tenDist, 1, True)
        tenDist = L.exp(tenDist)

        tenDivisor = L.reduce_sum(tenDist, 1, True)
        tenDivisor = L.reciprocal(tenDivisor)

        tenScaleX = L.unfold(x=tenFlow[:, 0:1, :, :],
                             kernel_sizes=self.intUnfold,
                             strides=1,
                             paddings=int((self.intUnfold - 1) / 2))  # [b, c, h * w]
        tenScaleX = L.reshape(tenScaleX, (b, -1, h, w))  # [b, c, h, w]
        tenScaleX = self.moduleScaleX(tenDist * tenScaleX) * tenDivisor

        tenScaleY = L.unfold(x=tenFlow[:, 1:2, :, :],
                             kernel_sizes=self.intUnfold,
                             strides=1,
                             paddings=int((self.intUnfold - 1) / 2))  # [b, c, h * w]
        tenScaleY = L.reshape(tenScaleY, (b, -1, h, w))  # [b, c, h, w]
        tenScaleY = self.moduleScaleY(tenDist * tenScaleY) * tenDivisor

        return L.concat([tenScaleX, tenScaleY], 1)


class Network(dg.Layer):

    def __init__(self, ):
        super(Network, self).__init__()

        self.moduleFeatures = Features()
        levels = [2, 3, 4, 5, 6]
        self.moduleMatching = dg.LayerList([Matching(intLevel) for intLevel in levels])
        self.moduleSubpixel = dg.LayerList([Subpixel(intLevel) for intLevel in levels])
        self.moduleRegularization = dg.LayerList([Regularization(intLevel) for intLevel in levels])

    def forward(self, tenFirst, tenSecond):
        tenFeaturesFirst = self.moduleFeatures(tenFirst)
        tenFeaturesSecond = self.moduleFeatures(tenSecond)

        tenFirst = [tenFirst]
        tenSecond = [tenSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            h, w = tenFeaturesFirst[intLevel].shape[2:]
            tenFirst.append(L.image_resize(tenFirst[-1], out_shape=(h, w), align_corners=False))
            tenSecond.append(L.image_resize(tenSecond[-1], out_shape=(h, w), align_corners=False))

        tenFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.moduleMatching[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                        tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel],
                                                        tenFlow)
            tenFlow = self.moduleSubpixel[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                        tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel],
                                                        tenFlow)
            tenFlow = self.moduleRegularization[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                              tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel],
                                                              tenFlow)

        return tenFlow * 20.0
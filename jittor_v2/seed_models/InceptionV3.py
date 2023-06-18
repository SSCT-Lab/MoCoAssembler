import jittor
import jittor as jt
import jittor.nn as nn


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,
                  padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=80, kernel_size=3, stride=1),
            ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        )


        self.block3_1 = InceptionV3ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32)
        self.block3_2 = InceptionV3ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64)
        self.block3_3 = InceptionV3ModuleA(in_channels=288, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64)



        self.block4_1 = InceptionV3ModuleD(in_channels=288, out_channels1reduce=384, out_channels1=384, out_channels2reduce=64, out_channels2=96)
        self.block4_2 = InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=128, out_channels2=192, out_channels3reduce=128, out_channels3=192, out_channels4=192)
        self.block4_3 = InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192)
        self.block4_4 = InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192)
        self.block4_5 = InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=192, out_channels2=192, out_channels3reduce=192, out_channels3=192, out_channels4=192)


        self.aux_logits = InceptionAux(in_channels=768, out_channels=num_classes)


        self.block5_1 = InceptionV3ModuleE(in_channels=768, out_channels1reduce=192, out_channels1=320, out_channels2reduce=192, out_channels2=192)
        self.block5_2 = InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192)
        self.block5_3 = InceptionV3ModuleC(in_channels=2048, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192)


        self.max_pool = jittor.nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = jittor.nn.Dropout(p=0.5)
        self.linear = jittor.nn.Linear(2048, 1000)

    def execute(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block4_4(x)
        x = self.block4_5(x)
        aux = x
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)


        aux = self.aux_logits(aux)
        return x


class InceptionV3ModuleA(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=5, padding=2)
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
        )

    def execute(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = jt.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleB(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=(1, 7), paddings=(0, 3)),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(7, 1), paddings=(3, 0))
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=(1, 7), paddings=(0, 3)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=(7, 1), paddings=(3, 0)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=(1, 7), paddings=(0, 3)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=(7, 1), paddings=(3, 0))
        )

        self.branch4 = nn.Sequential(
            jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
        )

    def execute(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = jt.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleC(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(1, 3), paddings=(0, 1))
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(3, 1), paddings=(1, 0))

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=(3, 1), paddings=(1, 0))
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=(1, 3), paddings=(0, 1))

        self.branch4 = nn.Sequential(
            jittor.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
        )

    def execute(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        x21 = self.branch2_conv2a(x2)
        x22 = self.branch2_conv2b(x2)
        out2 = jt.cat([x21, x22], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        x31 = self.branch3_conv3a(x3)
        x32 = self.branch3_conv3b(x3)
        out3 = jt.cat([x31, x32], dim=1)
        out4 = self.branch4(x)
        out = jt.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def execute(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = jt.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=(1, 7), paddings=(0, 3)),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=(7, 1), paddings=(3, 0)),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2)
        )

        self.branch3 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)

    def execute(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = jt.cat([out1, out2, out3], dim=1)
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = jittor.nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = jittor.nn.Conv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.auxiliary_dropout = jittor.nn.Dropout(p=0.7)
        self.auxiliary_linear1 = jittor.nn.Linear(in_features=768, out_features=out_channels)

    def execute(self, x):
        x = self.auxiliary_avgpool(x)
        x = self.auxiliary_conv1(x)
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        x = self.auxiliary_dropout(x)
        out = self.auxiliary_linear1(x)
        return out




def go():
    model = InceptionV3()
    x = jt.randn((3, 3, 299, 299))
    y = model(x)
    print(y)

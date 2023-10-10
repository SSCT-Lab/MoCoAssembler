"""
@Fire
https://github.com/fire717
"""

import numpy as np

import paddle
import paddle.nn as nn




"""
nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1,  ))
"""

def conv_3x3_act(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU()
    )


def conv_1x1_act(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU()
    )

def conv_1x1_act2(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU()
    )



def dw_conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=False),
        nn.BatchNorm2D(inp),
        nn.ReLU(),
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup)
    )

def dw_conv2(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=False),
        nn.BatchNorm2D(inp),
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU(),
    )

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=False),
        nn.BatchNorm2D(inp),
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.ReLU(),
    )



def upsample(inp, oup, scale=2):
    return nn.Sequential(
                nn.Conv2D(inp, inp, 3, 1, 1, groups=inp),
                nn.ReLU(),
                conv_1x1_act2(inp,oup),
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))



def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape((batchsize, groups,
               channels_per_group, height, width))

    x = x.transpose([0, 2, 1, 3, 4])
    # x = paddle.transpose(x, (1, 2)).contiguous()

    # flatten
    x = x.reshape((batchsize, -1, height, width))

    return x


class InvertedResidual(nn.Layer):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2D(inp),
                nn.Conv2D(inp, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(branch_features),
                nn.ReLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2D(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2D(branch_features),
            nn.Conv2D(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2D:
        return nn.Conv2D(i, o, kernel_size, stride, padding, bias_attr=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, axis=1)
            out = paddle.concat((x1, self.branch2(x2)), axis=1)
        else:
            out = paddle.concat((self.branch1(x), self.branch2(x)), axis=1)

        out = channel_shuffle(out, 2)

        return out

#0.5:[4, 8, 4], [24, 48, 96, 192, 1024]
#1.0:[4, 8, 4], [24, 116, 232, 464, 1024]
class Backbone(nn.Layer):
    def __init__(
        self,
        stages_repeats=[4, 8, 4], 
        stages_out_channels=[24, 64, 128, 192, 256],
        num_classes = 1000,
        inverted_residual = InvertedResidual
        ) -> None:
        super(Backbone, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channels, output_channels, 3, 2, 1, bias_attr=False),
            nn.BatchNorm2D(output_channels),
            nn.ReLU(),
        )
        input_channels = output_channels

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # output_channels = self._stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2D(input_channels, output_channels, 1, 1, 0, bias_attr=False),
        #     nn.BatchNorm2D(output_channels),
        #     nn.ReLU(),
        # )



        self.upsample2 = upsample(stages_out_channels[3], stages_out_channels[2])
        self.upsample1 = upsample(stages_out_channels[2], stages_out_channels[1])

        self.conv3 = nn.Conv2D(stages_out_channels[2], stages_out_channels[2], 1, 1, 0)
        self.conv2 = nn.Conv2D(stages_out_channels[1], stages_out_channels[1], 1, 1, 0)

        self.conv4 = dw_conv3(stages_out_channels[1], 24, 1)


    def _forward_impl(self, x):
        # See note [paddleScript super()]
        x = x/127.5-1
        
        x = self.conv1(x)
        # x = self.maxpool(x)
        f1 = self.stage2(x)
        #print(f1.shape)#2, 116, 24, 24]
        f2 = self.stage3(f1)
        #print(f2.shape)#2, 232, 12, 12]
        x = self.stage4(f2)
        #print(x.shape)#2, 464, 6, 6]

        x = self.upsample2(x)
        f2 = self.conv3(f2)
        x += f2

        x = self.upsample1(x)
        f1 = self.conv2(f1)
        x += f1

        x = self.conv4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)




class HardSigmoid(nn.Layer):
    """Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_
    This activation is computed as follows:
    .. math::
        f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2))
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * (x/(1+paddle.abs(x)))+0.5

class Header(nn.Layer):
    def __init__(self, num_classes, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        #heatmaps, centers, regs, offsets
        #Person keypoint heatmap
        self.header_heatmaps = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes, 1, 1, 0),
                        # nn.Sigmoid(),
                        HardSigmoid(),
                    ])

        #Person center heatmap
        self.header_centers = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, 1, 1, 1, 0),
                        # nn.Sigmoid(),
                        HardSigmoid(),
                        # MulReshapeArgMax()
                    ])

        #Keypoint regression field:
        self.header_regs = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0),
                    ])

        #2D per-keypoint offset field
        self.header_offsets = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0),
                    ])

    def argmax2loc(self, x, h=48, w=48):
        ## n,1
        y0 = paddle.div(x,w).long()
        x0 = paddle.sub(x, y0*w).long()
        return x0,y0


    def forward(self, x):

        res = []
        if self.mode=='train':
            h1 = self.header_heatmaps(x)
            h2 = self.header_centers(x)
            h3 = self.header_regs(x)
            h4 = self.header_offsets(x)
            res = [h1,h2,h3,h4]


        elif self.mode=='test':
            pass


        else:
            print("[ERROR] wrong mode.")

        

        return res


class MoveNet(nn.Layer):
    def __init__(self, num_classes=7, width_mult=1.,mode='train'):
        super(MoveNet, self).__init__()

        self.backbone = Backbone()

        self.header = Header(num_classes, mode)
        

        self._initialize_weights()


    def forward(self, x):
        x = self.backbone(x) # n,24,48,48
        # print(x.shape)

        x = self.header(x)
        # print([x0.shape for x0 in x])

        return x


    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # m.weight.data.normal_(0, 0.01)
                attr_p = paddle.ParamAttr(initializer=nn.initializer.KaimingNormal())
                m.weight = paddle.create_parameter(m.weight.shape, dtype=m.weight.dtype, attr= attr_p)
                # paddle.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # paddle.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
                    # nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
                # m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()




if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = MoveNet()
    print(paddle.summary(model, (1, 3, 192, 192)))


    dummy_input1 = paddle.randn((1, 3, 192, 192))
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    
    # paddle.onnx.export(model, dummy_input1, "pose.onnx", 
    #     verbose=True, input_names=input_names, output_names=output_names,
    #     do_constant_folding=True, opset_version=11)
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


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, n):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.n = n


        self.conv1 = nn.Sequential(
            # pw
            nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(hidden_dim),
            nn.ReLU(),
            # dw
            nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim,  ),
            nn.BatchNorm2D(hidden_dim),
            nn.ReLU(),
            # pw-linear
            nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(oup),
        )

        self.conv2 = nn.Sequential(
            # pw
            nn.Conv2D(oup, hidden_dim, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(hidden_dim),
            nn.ReLU(),
            # dw
            nn.Conv2D(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias_attr=False),
            nn.BatchNorm2D(hidden_dim),
            nn.ReLU(),
            # pw-linear
            nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(oup),
        )

    def forward(self, x):
        x = self.conv1(x)

        for _ in range(self.n):
            x = x + self.conv2(x)

        return x



# class MulReshapeArgMax(nn.Layer):
#     def __init__(self):
#         super(MulReshapeArgMax, self).__init__()


#         self.weight = torch.reshape(weight, (1,48,48))

#     def forward(self, x):
#         # x n,1,48,48
#         #print(self.weight.shape)
#         x = x.mul_(self.weight)
#         x = torch.reshape(x, (-1,2304))
#         x = torch.argmax(x, dim=1, keepdim=True)

#         return x
    



class Backbone(nn.Layer):
    def __init__(self):
        super(Backbone, self).__init__()
        #mobilenet v2


        input_channel = 32

        self.features1 = nn.Sequential(*[
                            conv_3x3_act(3, input_channel, 2),
                            dw_conv(input_channel, 16, 1),
                            InvertedResidual(16, 24, 2, 6, 1)
                        ])

        self.features2 = InvertedResidual(24, 32, 2, 6, 2)
        self.features3 = InvertedResidual(32, 64, 2, 6, 3)

        self.features4 = nn.Sequential(*[
                            InvertedResidual(64, 96, 1, 6, 2),
                            InvertedResidual(96, 160, 2, 6, 2),
                            InvertedResidual(160, 320, 1, 6, 0),
                            conv_1x1_act(320,1280),
                            nn.Conv2D(1280, 64, 1, 1, 0, bias_attr=False),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                        ])


        self.upsample2 = upsample(64, 32)
        self.upsample1 = upsample(32, 24)

        self.conv3 = nn.Conv2D(64, 64, 1, 1, 0)
        self.conv2 = nn.Conv2D(32, 32, 1, 1, 0)
        self.conv1 = nn.Conv2D(24, 24, 1, 1, 0)

        self.conv4 = dw_conv3(24, 24, 1)



    def forward(self, x):
        x = x/127.5-1


        f1 = self.features1(x)
        #print(f1.shape)#1, 24, 48, 48]
        
        f2 = self.features2(f1)
        #print(f2.shape)#1, 32, 24, 24]

        f3 = self.features3(f2)
        #print(f3.shape)#[1, 64, 12, 12]

        f4 = self.features4(f3)
        f3 = self.conv3(f3)
        #print(f4.shape)#[1, 64, 12, 12]
        f4 += f3

        f4 = self.upsample2(f4)
        f2 = self.conv2(f2)
        f4 += f2

        f4 = self.upsample1(f4)
        f1 = self.conv1(f1)
        f4 += f1

        f4 = self.conv4(f4)

        return f4



class Header(nn.Layer):
    def __init__(self, num_classes, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        #heatmaps, centers, regs, offsets
        #Person keypoint heatmap
        self.header_heatmaps = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes, 1, 1, 0,  ),
                        nn.Sigmoid()
                    ])

        #Person center heatmap
        self.header_centers = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, 1, 1, 1, 0,  ),
                        nn.Sigmoid(),
                        # MulReshapeArgMax()
                    ])

        #Keypoint regression field:
        self.header_regs = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0,  ),
                    ])

        #2D per-keypoint offset field
        self.header_offsets = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2D(96, num_classes*2, 1, 1, 0,  ),
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

        elif self.mode=='all':
            #print("Header: ", x.shape)
            pass
            # h1 = self.head1(x)
            # # print(h1.shape)     # n,24,48,48

            # h2 = self.head2(x)  # n,1
            # # print(h2.shape,h2)
            # x0,y0 = self.argmax2loc(h2) #n,1
            # # print(x0,y0,x0.shape,y0.shape)

            # # h2_1 = torch.cat([y0,x0], -1).long().unsqueeze(1)
            # #torch.Tensor([[0] for _ in range(x.shape[0])]),
            # # print(h2_1.shape, h2_1) # n,2

            # h3 = self.head3(x)  # n,34,48,48
            # h2_1 = []
            # for i in range(x.shape[0]):
            #     #print(h3[i,:,y0[i],x0[i]].shape)
            #     h2_1.append(h3[i,:,y0[i],x0[i]])
            # h2_1 = torch.cat(h2_1, dim=1).transpose(1,0).reshape((-1,17,2))
            # #print(h2_1.shape) # n,17,2

            # h2_2 = h2_1[:,:,0].add_(y0) #n,17
            # #print(h2_2.shape)

            # h2_2_tmp = []
            # # print(self.range_tensor.shape) #48,48,17
            # for i in range(x.shape[0]):
            #     h2_2_t = self.range_tensor.sub_(h2_2[i])
            #     print(h2_2_t.shape)
            #     h2_2_tmp.append(h2_2_t)
            # h2_2 = torch.cat(h2_2_tmp, dim=0)
            # print(h2_2.shape)

            # h2_1 = h2_1[:,:,1].add_(x0)


            # # h2_1 = h3[h2_1]
            # # h2_1 = torch.index_select(h3, 0, h2_1)
            # print(h2_1.shape) #n,17

            # h4 = self.head4(x)  # n,34,48,48
            # # print(h4.shape)
            # b
        else:
            print("[ERROR] wrong mode.")

        

        return res


class MoveNet(nn.Layer):
    def __init__(self, num_classes=17, width_mult=1.,mode='train'):
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
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()




if __name__ == "__main__":

    model = MoveNet()
    print(paddle.summary(model, (1, 3, 192, 192)))


    dummy_input1 = paddle.randn((1, 3, 192, 192))
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    
    # torch.onnx.export(model, dummy_input1, "pose.onnx", 
    #     verbose=True, input_names=input_names, output_names=output_names,
    #     do_constant_folding=True,opset_version=11)
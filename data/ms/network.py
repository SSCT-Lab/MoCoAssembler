import mindspore
import mindspore.nn as nn
import numpy as np
import random
from hs_data import load_all_sr,load_camera
class conv_block(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,has_bias=True,pad_mode='pad'),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,has_bias=True,pad_mode='pad'),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )


    def construct(self,x):
        x = self.conv(x)
        return x

class filter_block(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(filter_block,self).__init__()
        self.conv =nn.Conv2d(ch_in,1020,1,stride=1,padding=0,has_bias=False )
        a = load_all_sr()
        #a=load_camera(25)
        t = mindspore.Tensor(a)
        self.conv.weight = mindspore.Parameter(t)
        self.conv.weight.requires_grad = True
        #at = np.random.normal(loc=0.5,size=(1, 1020, 1, 1)).astype(np.float32)

        at = np.zeros((1, 1020, 1, 1)).astype(np.float32)
        for i in range(100):
            n = random.randint(0, 1019)
            at[0, n, 0, 0] =random.randint(0,10)/1000
        #print(at)
        #at[0,454,0,0]=3.55e-03
        #at[0, 262, 0, 0] = 2.52E-03
        #at[0, 842, 0, 0] = 1.89E-03
        # at[0, 609, 0, 0] = 1.61E-03
        # at[0, 943, 0, 0] = 1.38E-03


        self.at = mindspore.Tensor(at)

        #self.at = mindspore.Parameter(self.at)
        self.at.requires_grad=False
        #self.Conv_1x1 = nn.Conv2d(1020, 3, kernel_size=1, stride=1, padding=0,has_bias=False)
        #self.Conv_1x1.weight.data.zero_()
        self.conv2 = nn.Conv2d(1020, 64, 1, stride=1, padding=0)
    def construct(self,x):
        x = self.conv(x)
        #self.at=mindspore.sigmoid(self.at)
        at=self.at#*(self.at>0.0071)
        #x=x[:,]
        #x= (x*self.at)[:,idx[0:10]]
        #x=self.Conv_1x1(x)
        x=self.conv2(x)
        return x,at#self.Conv_1x1.weight

class up_conv(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up=nn.ResizeBilinear()
        self.conv = nn.SequentialCell(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,has_bias=True,pad_mode='pad'),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )

    def construct(self,x):
        x = self.up(x,scale_factor=2)
        x=self.conv(x)
        return x

class Recurrent_block(nn.Cell):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.SequentialCell(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,has_bias=True,pad_mode='pad'),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )

    def construct(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Cell):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.SequentialCell(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def construct(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def construct(self,x):
        x = self.conv(x)
        return x

class FS_Net_RGB(nn.Cell):
    def __init__(self, img_ch=3, output_ch=31):
        super(FS_Net_RGB, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.cat=mindspore.ops.Concat(axis=1)

    def construct(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = self.cat((x4, d5))

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat((x3, d4))
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat((x2, d3))
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat((x1, d2))
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class FS_Net(nn.Cell):
    def __init__(self, img_ch=1020, output_ch=1):
        super(FS_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = filter_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.cat=mindspore.ops.Concat(axis=1)

    def construct(self, x):
        # encoding path
        x1,at = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = self.cat((x4, d5))

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat((x3, d4))
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat((x2, d3))
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat((x1, d2))
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

if __name__=="__main__":
    net=FS_Net()
    shape=(1,31,128,128)
    uniformreal =mindspore.ops.Ones()

    im=uniformreal(shape,mindspore.float32)
    out=net(im)
    print(out)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
def setdir():
    fapath = os.path.dirname(os.getcwd())
    sys.path.append(fapath)
setdir()

import numpy as np
import paddle.fluid as fluid
from utils.modules import ConvBnReLu,Bottleneck
class AdaptivePool(fluid.dygraph.Layer):
    def __init__(self, size):
        super(AdaptivePool, self).__init__()
        self.size = (size, size)
    def forward(self, x):
        x = fluid.layers.adaptive_pool2d(x, self.size, "max")
        return x

class PSPModule(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = fluid.dygraph.LayerList([self.diff_pool(in_channels, out_channels, size) for size in sizes])
        self.bottleneck = ConvBnReLu(in_channels+len(sizes)*out_channels, out_channels, 3, padding=1)

    '''将特征图通过不同大小的池化并且upsample到同一大小'''
    def diff_pool(self, in_channels, out_channels, size):
        prior = AdaptivePool(size)
        conv = fluid.dygraph.Conv2D(in_channels, out_channels, filter_size=1, bias_attr=False)
        bn = fluid.dygraph.BatchNorm(out_channels)
        return fluid.dygraph.Sequential(prior, conv, bn)
    
    def forward(self, feature_map):
        h, w = feature_map.shape[2], feature_map.shape[3]
        priors = [fluid.layers.interpolate(input=stage(feature_map), out_shape=(h, w)) for stage in self.stages] + [feature_map]
        priors = fluid.layers.concat(priors, axis=1)
        bottle = self.bottleneck(priors)
        bottle = fluid.layers.dropout(bottle, 0.1)
        return bottle

class PSPNet(fluid.dygraph.Layer):

    def __init__(self,num_classes,layers =[3,4,6,3]):
        super(PSPNet, self).__init__()
        self.in_channels = 128
        self.convblock = fluid.dygraph.Sequential(
            ConvBnReLu(3,64,3,2,1),
            ConvBnReLu(64,64,3,1,1),
            ConvBnReLu(64,128,3,1,1),
        )
        self.maxpool = fluid.dygraph.Pool2D(pool_size=3,pool_stride=2,pool_padding=1)
        self.layer1 = self.make_layer(layers[0], 64)
        self.layer2 = self.make_layer(layers[1], 128, stride = 2)
        self.layer3 = self.make_layer(layers[2], 256, stride = 1, dilation = 2)
        self.layer4 = self.make_layer(layers[3], 512, stride = 1, dilation = 4)
        self.out = ConvBnReLu(512,num_classes,3,1,1)
        self.au_out = ConvBnReLu(512, num_classes,3, 1, 1)
        self.head = PSPModule(2048, 512)

    '''resnet的4个块'''
    def make_layer(self, blocks, out_channels, stride =1, dilation = 1):
        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, dilation))
        '''第一层之后的 in_channels = out_channels'''
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, dilation = dilation))
        return fluid.dygraph.Sequential(*layers)

    def forward(self, input):
        x = self.convblock(input)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature_map = self.layer2(x)
        x = self.layer3(feature_map)
        x = self.layer4(x)
        x = self.head(x)
        feature_map = fluid.layers.image_resize(feature_map,(input.shape[2], input.shape[3]))
        x = fluid.layers.image_resize(x,(input.shape[2], input.shape[3]))
        x = self.out(x)
        x = fluid.layers.softmax(x,axis = 1)

        au_x = self.au_out(feature_map)
        au_x = fluid.layers.softmax(x,axis = 1)
        return x, au_x


'''测试网络'''
if __name__ == "__main__":
    with fluid.dygraph.guard():
        img = np.zeros([1,3,512,512]).astype('float32')
        model = PSPNet(3)
        img = fluid.dygraph.to_variable(img)
        outs,au_out= model(img)
        au_outs = fluid.layers.transpose(outs,perm=(0,2,3,1))
        print(au_outs.numpy()[0][0][0])
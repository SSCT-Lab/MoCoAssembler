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
from config import IMG_WEIGHT, IMG_HEIGHT

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

    '''resnet的4个块'''
    def make_layer(self, blocks, out_channels, stride =1, dilation = 1):
        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, dilation))
        '''第一层之后的 in_channels = out_channels'''
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, dilation = dilation))
        return fluid.dygraph.Sequential(*layers)


    '''将特征图通过不同大小的池化并且upsample到同一大小'''
    def diff_pool(self, feature_map, out_channels, size):

        h, w = feature_map.shape[2], feature_map.shape[3]
        pool_map = fluid.layers.adaptive_pool2d(feature_map, pool_size=size, pool_type="max")
        # print("pool_map:",pool_map.shape)
        pool_map = fluid.layers.conv2d(pool_map, out_channels, filter_size= 1, groups=1, bias_attr=False)
        pool_map = fluid.layers.batch_norm(pool_map)
        pool_map = fluid.layers.image_resize(pool_map, out_shape=(h,w))
        # print("pool_map:",pool_map.shape)
        return pool_map
    
    '''金字塔池化层'''
    def pyramid_pooling(self, feature_map, out_channels, sizes = (1,2,3,6)):

        diff_pools = [self.diff_pool(feature_map,out_channels,size) for size in sizes] + [feature_map]
        cat_pool = fluid.layers.concat(diff_pools, axis = 1)
        # print("cat_pool:",cat_pool.shape)
        cat_pool = fluid.layers.conv2d(cat_pool, out_channels, filter_size=3, padding=1, dilation = 1,groups=1, bias_attr=False)
        cat_pool = fluid.layers.batch_norm(cat_pool)
        cat_pool = fluid.layers.relu(cat_pool)
        cat_pool = fluid.layers.dropout(cat_pool,0.1)
        return cat_pool

    def forward(self,input):
        x = self.convblock(input)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x,512)
        x = fluid.layers.image_resize(x,(input.shape[2],input.shape[3]))
        x = self.out(x)
        x = fluid.layers.softmax(x,axis = 1)
        return x


'''测试网络'''
if __name__ == "__main__":
    with fluid.dygraph.guard():
        img = np.zeros([1,3,512,512]).astype('float32')
        model = PSPNet(3)
        img = fluid.dygraph.to_variable(img)
        outs = model(img)
        outs = fluid.layers.transpose(outs,perm=(0,2,3,1))
        print(outs.numpy()[0][0][0])
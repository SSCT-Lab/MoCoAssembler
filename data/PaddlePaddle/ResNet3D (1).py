import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Pool2D, BatchNorm, Linear ,Conv3D    #Conv2D, 
from paddle.fluid.dygraph.base import to_variable

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size =7,
                 stride=1,
                 padding =0,
                 groups=1,
                 act=None,
                 ):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv3D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)     ##这是2D的BatchNorm，后面需要注意

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            padding = 0,
            act=None)   ####act='relu'
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            padding = 1,
            act=None)    ###act='relu'
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters= num_filters*4  ,     #####  通道*4倍
            # num_filters= num_filters  ,       
            filter_size=1,
            padding = 0,
            act='relu')

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_channels=num_channels,
                # num_filters=num_filters ,
                num_filters= num_filters*4  ,  ##### 通道*4倍
                filter_size=3,
                stride=stride,
                padding =1 )

        self.shortcut = shortcut

        # self._num_channels_out = num_filters 
        self._num_channels_out = num_filters *4    #####  通道*4倍

    def forward(self, inputs):
        input1 = to_variable(inputs)
        # print(input1.shape,"bottleneck_in")    ####测试 
        y = self.conv0(inputs)
        
        conv1 = self.conv1(y)
        # print(conv1.shape)    ###测试
        conv2 = self.conv2(conv1)
        # print(conv1.shape)    ###测试

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        # print(y.shape,"bottleneck_out")     ####测试 

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return  layer_helper.append_activation(y)   ###    conv2 ## 

class ResNet3D(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=101, seg_num=10, weight_devay=None,conv1_t_size=7):
        super(ResNet3D, self).__init__(name_scope)

        self.layers = layers
        self.seg_num = seg_num
        self.conv1_t_size = conv1_t_size
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            self.full_name(),
            num_channels=self.seg_num,                ####self.seg_num就是图像输入的维度
            num_filters=64,
            filter_size=7, 
            stride=[1,2,2],
            padding =[self.conv1_t_size//2,3,3] ,                      
            act="relu")
        # self.pool3d_max = fluid.layers.pool3d(                 ####self.pool3d_max不需提前定义，提前定义反而会报错。
        #     pool_size=3,
        #     pool_stride=2,
        #     pool_padding=1,
        #     pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=[2,2,2] if i == 0 and block != 0 else 1,     #### stride=2 if i == 0 and block != 0 else 1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # self.pool3d_avg = fluid.layers.pool3d(pool_size=1, pool_type='avg', global_pooling=True)  ###定义会出错

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(input_dim=num_channels,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label=None):
        # out = fluid.layers.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        y = self.conv(inputs)
        # print(y.shape)  ####
        y = fluid.layers.pool3d(y,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        # print(y.shape,"---------------")   ####
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
            # print("---------------")   ####
        # print(y.shape)    ####
        y = fluid.layers.pool3d(y,pool_size=1, pool_type='avg', global_pooling=True)
        # print(y.shape)    ####

        # return y ####测试
        out = fluid.layers.reshape(x=y, shape=[-1,  y.shape[1]])
        # return out ####测试

        # out = fluid.layers.reduce_mean(out, dim=1)
        # return out ####测试
        y = self.out(out)

        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y

# if __name__ == '__main__':
#     with fluid.dygraph.guard():
#         network = ResNet3D('resnet', 50,seg_num=16,)
#         img = np.ones([5, 16, 13, 224, 224]).astype('float32')
#         img = fluid.dygraph.to_variable(img)
#         outs = network(img).numpy()
        
#         # print(network.state_dict().keys())   ###打印网络的层名称列表
#         # print(network.state_dict())
#         # print(outs,outs.shape)
#         print(outs.shape)

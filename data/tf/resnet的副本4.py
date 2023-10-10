#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: kaifang zhang
@license: Apache License
@time: 2020/12/01
@contact: 1115291605@qq.com
第2个版本  2020-08-19日  @zhangkaifang  更接近最真实的resnet18
"""


import tensorflow as tf
from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers
import tensorflow.keras as keras


#  定义一个3x3卷积！
def regularized_padded_conv(*args, **kwargs):
    '''
    kernel_initializer='glorot_normal': Glorot正态分布初始化方法，也称作Xavier正态分布初始化，
                        参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，其中
                        fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目;
    kernel_initializer='he_normal': He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in)
                        的正态分布产生，其中fan_in权重张量的扇入, kernel_regularizer=regularizers.l2(5e-4)
    '''
    return layers.Conv2D(*args, **kwargs, padding='same',
                         use_bias=False, kernel_initializer='glorot_normal')

# 1.定义 Basic Block 模块。对于Resnet18和Resnet34
class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 1. BasicBlock模块中的第1个卷积层;
        self.conv1 = regularized_padded_conv(out_channels, kernel_size=3, strides=stride)
        self.bn1 = layers.BatchNormalization()

        # 2. BasicBlock模块中的第2个卷积;第1个卷积如果做stride就会有一个下采样，在这个里面就不做下采样了。这一块始终保持size一致，把stride固定为1
        self.conv2 = regularized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        # 3. 判断stride是否等于1,如果为1就是没有降采样。
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels,
                                                                kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        # if training: print("=> training network ... ")
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


# 1.定义 Bottleneck 模块。对于Resnet50,Resnet101和Resnet152;
class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([layers.Conv2D(self.expansion * out_channels, kernel_size=1,
                                                      strides=strides, use_bias=False),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)

        out = out + self.shortcut(x, training)
        out = tf.nn.relu(out)

        return out


# 2. ResBlock 模块。继承keras.Model或者keras.Layer都可以
class ResNet(keras.Model):

    # 第1个参数layer_dims：[2, 2, 2, 2] 4个Res Block，每个包含2个Basic Block，
    # 第3参数num_classes：我们的全连接输出，取决于输出有多少类。
    def __init__(self, blocks, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 0. 预处理卷积层；实现比较灵活可以加MAXPool2D，或者不加，这里没加。
        # 注意这里的channels需要和layer1的channels是一样的，不然能add。
        self.stem = Sequential([regularized_padded_conv(64, kernel_size=3, strides=1),
                                layers.BatchNormalization()])

        # 1. 创建4个ResBlock；注意第1项不一定以2倍形式扩张，都是比较随意的，这里都是经验值。
        self.layer1 = self.build_resblock(blocks, 64,   layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(blocks, 128,  layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(blocks, 256,  layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(blocks, 512,  layer_dims[3], stride=2)
        # self.final_bn  = layers.BatchNormalization()

        # self.avgpool = layers.GlobalAveragePooling2D()
        # self.fc = layers.Dense(num_classes)

    # 2. 创建ResBlock
    def build_resblock(self, blocks, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)                    # [1]*3 = [1, 1, 1]
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(blocks(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks


    def call(self,inputs, training=False):
        # __init__中准备工作完毕；下面完成前向运算过程。
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        # out = self.final_bn(out, training=training)
        # out = tf.nn.relu(out)

        # 做一个global average pooling，得到之后只会得到一个channel，不需要做reshape操作了。
        # shape为 [batchsize, channel]
        # x = self.avgpool(x)
        # # [b, 100]
        # x = self.fc(x)

        return out


""" Resnet18 """
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

""" ResNet-34，那34是怎样的配置呢？只需要改一下这里就可以了。4个Res Block """
# 如果我们要使用
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

""" Resnet50 """
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

""" Resnet101 """
def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

""" Resnet152 """
def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


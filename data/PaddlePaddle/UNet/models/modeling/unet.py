# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import paddle
import paddle.fluid as fluid
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, bn_relu, relu
from models.libs.model_libs import conv, max_pool, deconv
from models.unet_backbone.resnet import ResNet as resnet_backbone
from models.unet_backbone.resnet_vd import ResNet as resnet_vd_backbone
from models.unet_backbone.se_resnet import SE_ResNet_vd as se_resent_backbone
from models.unet_backbone.resnet_acnet import ResNetACNet as resnet_acnet_backbone
from models.unet_backbone.vgg import VGGNet as vgg_backbone
from models.unet_backbone.hrnet import HRNet as hrnet_backbone



def double_conv(data, out_ch):
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.33))
    with scope("conv0"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1, param_attr=param_attr))
    with scope("conv1"):
        data = bn_relu(
            conv(data, out_ch, 3, stride=1, padding=1, param_attr=param_attr))
    return data


def down(data, out_ch):
    # 下采样：max_pool + 2个卷积
    with scope("down"):
        data = max_pool(data, 2, 2, 0)
        data = double_conv(data, out_ch)
    return data


def up(data, short_cut, out_ch):
    # 上采样：data上采样(resize或deconv), 并与short_cut concat
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.XavierInitializer(),
    )
    with scope("up"):
        if cfg.MODEL.UNET.UPSAMPLE_MODE == 'bilinear':
            data = fluid.layers.resize_bilinear(data, short_cut.shape[2:])
        else:
            data = deconv(
                data,
                out_ch // 2,
                filter_size=2,
                stride=2,
                padding=0,
                param_attr=param_attr)
        data = fluid.layers.concat([data, short_cut], axis=1)
        data = double_conv(data, out_ch)
    return data


def encode(data):
    # 编码器设置
    short_cuts = []
    with scope("encode"):
        with scope("block1"):
            data = double_conv(data, 64)
            short_cuts.append(data)
        with scope("block2"):
            data = down(data, 128)
            short_cuts.append(data)
        with scope("block3"):
            data = down(data, 256)
            short_cuts.append(data)
        with scope("block4"):
            data = down(data, 512)
            short_cuts.append(data)
        with scope("block5"):
            data = down(data, 512)
    return data, short_cuts


def decode(data, short_cuts):
    # 解码器设置，与编码器对称
    if cfg.MODEL.UNET.BACKBONE=="":
        short_shape=[256,128,64,64]
    else:
        short_shape=[cut.shape[1] for cut in short_cuts]
    with scope("decode"):
        with scope("decode1"):
            data = up(data, short_cuts[3], short_shape[3])
        with scope("decode2"):
            data = up(data, short_cuts[2], short_shape[2])
        with scope("decode3"):
            data = up(data, short_cuts[1], short_shape[1])
        with scope("decode4"):
            data = up(data, short_cuts[0], short_shape[0])
            
    return data

def resnet(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [18, 34, 50, 101, 152]:
        pass
    else:
        raise Exception("resnet only support layers in [18, 34, 50, 101, 152]")        
    model = resnet_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut

def resnet_vd(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [18, 34, 50, 101, 152,200]:
        pass
    else:
        raise Exception("resnet_vd only support layers in [18, 34, 50, 101, 152,200]")        
    model = resnet_vd_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut
def se_resent(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [18, 34, 50, 101, 152,200]:
        pass
    else:
        raise Exception("se_resent only support layers in [18, 34, 50, 101, 152,200]")        
    model = se_resent_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut

def resnet_acnet(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [18, 34, 50, 101, 152]:
        pass
    else:
        raise Exception("resnet_acnet only support layers in [18, 34, 50, 101, 152]")        
    model = resnet_acnet_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut

def hrnet(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [18, 30, 32, 40, 44, 48, 60, 64]:
        pass
    else:
        raise Exception("hrnet only support layers in [18, 30, 32, 40, 44, 48, 60, 64]")        
    model = hrnet_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut   

def vgg(input):
    layers= cfg.MODEL.UNET.LAYERS
    if layers in [11, 13, 16, 19]:
        pass
    else:
        raise Exception("vgg only support layers in [11, 13, 16, 19]")        
    model = vgg_backbone(layers)
    decode_shortcuts = model.net(input)
    data,decode_shortcut=decode_shortcuts[-1],decode_shortcuts[:-1]
    return data, decode_shortcut

def get_logit(data, num_classes,shape):
    # 根据类别数设置最后一个卷积层输出
    param_attr = fluid.ParamAttr(
        name='weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    with scope("logit"):
        data = conv(
            data, num_classes, 3, stride=1, padding=1, param_attr=param_attr)
        data = fluid.layers.resize_bilinear(data, shape)
    return data

#from model.unet_backbone import resnet,resnet_vd,hrnet,se_resent,vgg,resnet_acnet
def unet(input, num_classes):
    # UNET网络配置，对称的编码器解码器
    # Backbone设置：xception 或 mobilenetv2
    if cfg.MODEL.UNET.BACKBONE=="":
        encode_data, short_cuts = encode(input)

    elif cfg.MODEL.UNET.BACKBONE =='resnet':
        encode_data, short_cuts = resnet(input)

    elif cfg.MODEL.UNET.BACKBONE == 'resnet_vd':
        encode_data, short_cuts  = resnet_vd(input)

    elif  cfg.MODEL.UNET.BACKBONE =='se_resent':
        encode_data, short_cuts  = se_resent(input)

    elif cfg.MODEL.UNET.BACKBONE == 'resnet_acnet':
        encode_data, short_cuts  = resnet_acnet(input)

    elif cfg.MODEL.UNET.BACKBONE == 'hrnet':
        encode_data, short_cuts  = hrnet(input)

    elif cfg.MODEL.UNET.BACKBONE == 'vgg':
        encode_data, short_cuts  = vgg(input)
 
    else:
        raise Exception(
            "unet only support resnet, resnet_vd,resnet_acnet,hrnet,vgg and se_resent backbone")
    decode_data = decode(encode_data, short_cuts)
    shape=input.shape[2:]
    logit = get_logit(decode_data, num_classes,shape)
    return logit


if __name__ == '__main__':
    image_shape = [-1, 3, 320, 320]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    logit = unet(image, 4)
    print("logit:", logit.shape)
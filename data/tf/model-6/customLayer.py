import tensorflow as tf
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, BatchNormalization, Multiply, GlobalAveragePooling2D, \
    DepthwiseConv2D, Add, GlobalMaxPooling2D, Concatenate, ReLU, Dropout, SeparableConv2D

def _depth(filters, multiplier=1.0, base=8):
    round_half_up=int(int(filters) * multiplier / base+0.5)
    result=int(round_half_up * base)
    return max(result, base)

class ReLU6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.relu6(inputs)

class HSigmoid6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.relu6(inputs+np.float32(3)) * np.float32(1. / 6.)

class HSwish6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs * tf.nn.relu6(inputs+np.float32(3)) * np.float32(1. / 6.)

class Sigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.sigmoid(inputs)

def _Conv(inputs, filters, kernel_size=3, strides=2, padding='same', 
        normalization=BatchNormalization, activation=ReLU6,
        prefix=None, **conf_dict):

    x=Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding=padding,
              name=prefix+'Conv',
              **conf_dict)(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'BN')(x)

    if activation is not None:
        x=activation(name=prefix+'AC')(x)

    return x

def _DeptwiseConv(inputs, kernel_size=3, strides=2, padding='same', dilation_rate=1,
                normalization=BatchNormalization, activation=ReLU6,
                prefix=None, **conf_dict):

    conf_dict_inner = conf_dict.copy()

    if 'kernel_initializer' in  conf_dict_inner.keys():
        conf_dict_inner['depthwise_initializer'] = conf_dict_inner['kernel_initializer']
        conf_dict_inner.pop('kernel_initializer')
    if 'kernel_regularizer' in  conf_dict_inner.keys():
        #conf_dict_inner['depthwise_regularizer'] = conf_dict_inner['kernel_regularizer']
        conf_dict_inner.pop('kernel_regularizer')
    if 'use_bias' in  conf_dict_inner.keys():
        conf_dict_inner['use_bias'] = False
    else:
        if normalization is not None:
            conf_dict_inner['use_bias'] = False
        else:
            conf_dict_inner['use_bias'] = True

    x=DepthwiseConv2D(kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    name=prefix+'DepwiseConv',
                    **conf_dict_inner)(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'DepwiseBN')(x)

    if activation is not None:
        x=activation(name=prefix+'DepwiseAC')(x)

    return x

def _SeparableConv(inputs, filters, kernel_size=3, strides=2, padding='same', normalization=BatchNormalization, activation=ReLU6, prefix=None, **conf_dict):
    conf_dict_inner = conf_dict.copy()

    if 'kernel_initializer' in  conf_dict_inner.keys():
        conf_dict_inner['pointwise_initializer'] = conf_dict_inner['kernel_initializer']
        conf_dict_inner['depthwise_initializer'] = conf_dict_inner['kernel_initializer']
        conf_dict_inner.pop('kernel_initializer')
    if 'kernel_regularizer' in  conf_dict_inner.keys():
        conf_dict_inner['pointwise_regularizer'] = conf_dict_inner['kernel_regularizer']
        conf_dict_inner.pop('kernel_regularizer')

    x=SeparableConv2D(filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            **conf_dict_inner,
            name=prefix+'SepConv')(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'BN')(x)

    if activation is not None:
        x=activation(name=prefix+'AC')(x)

    return x

def _SEBlock(inputs, se_ratio, prefix, activation, **conf_dict):
    '''
        Reference:
                "Squeeze-and-Excitation Networks (CVPR 2018)"
                "Searching for MobileNetV3 (ICCV 2019)"
    '''
    infilters=backend.int_shape(inputs)[-1]
    conf_dict_inner = conf_dict.copy()
    if 'use_bias' in  conf_dict_inner.keys():
        conf_dict_inner['use_bias'] = True

    x=GlobalAveragePooling2D(keepdims=True, name=prefix+'SEAvgPool')(inputs)

    x=_Conv(x, filters=_depth(infilters*se_ratio), kernel_size=1, padding='valid',
        normalization=None, activation=activation, prefix=prefix+'SE_1', **conf_dict_inner)
    x=_Conv(x, filters=infilters, kernel_size=1, padding='valid',
        normalization=None, activation=HSigmoid6, prefix=prefix+'SE_2', **conf_dict_inner)

    return Multiply(name=prefix+'SEMul')([inputs, x])

def _IBN(x, expansion, filters, kernel_size=3, strides=1, dilation_rate=1, activation=ReLU6, attentionMode="_SEBlock", block_id=0, Residual=True, Detour=False, **conf_dict):
    shortcut=x
    infilters=backend.int_shape(x)[-1]

    prefix='IBN{}/'.format(block_id)

    if expansion > 1:
        x=_Conv(x, filters=_depth(infilters*expansion), kernel_size=1, 
                strides=1, activation=activation, prefix=prefix+'Expand', **conf_dict)
        out=x
    
    x=_DeptwiseConv(x, kernel_size=kernel_size, dilation_rate=dilation_rate, activation=activation,
                    strides=strides, prefix=prefix, **conf_dict)

    if attentionMode == '_SEBlock':
        x=_SEBlock(x, 0.25, prefix, activation, **conf_dict)

    x=_Conv(x, filters=filters, kernel_size=1, strides=1, activation=None, prefix=prefix+'Project', **conf_dict)

    
    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        return Add(name=prefix+'Add')([shortcut, x])
    else:
        if Detour:
            return x, out
        else:
            return x

def _Fused(x, expansion, filters, kernel_size=3, strides=1, activation=ReLU6, attentionMode="_SEBlock", block_id=0, Residual=True, **conf_dict):
    """Fused convolution layer."""
    shortcut=x
    infilters=backend.int_shape(x)[-1]
    prefix='FUC{}/'.format(block_id)

    x = _Conv(x,
            filters=_depth(infilters*expansion),
            kernel_size=kernel_size,
            strides=strides,
            activation_fn=activation,
            prefix=prefix+'Conv1',
            **conf_dict)
    out=x

    if attentionMode == '_SEBlock':
        x=_SEBlock(x, 0.25, prefix, activation, **conf_dict)

    x = _Conv(x,
              filters=filters,
              kernel_size=1,
              strides=1,
              activation_fn=None,
              prefix=prefix+'Conv2',
              **conf_dict)
    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        return x + shortcut
    else:
        return x

def _Tucker(x,
            input_rank_ratio=0.25,
            output_rank_ratio=0.25,
            filters=3,
            kernel_size=3,
            strides=1,
            activation=ReLU6,
            block_id=0,
            Residual=True,
            **conf_dict):

    shortcut=x
    infilters=backend.int_shape(x)[-1]
    prefix='TUC{}/'.format(block_id)

    x = _Conv(x,
            filters=_depth(infilters, input_rank_ratio),
            kernel_size=1,
            strides=1,
            activation_fn=activation,
            prefix=prefix+'Conv1',
            **conf_dict)
    x = _Conv(x,
            _depth(filters, output_rank_ratio),
            kernel_size=kernel_size,
            strides=strides,
            activation_fn=activation,
            prefix=prefix+'Conv2',
            **conf_dict)
    x = _Conv(x,
            filters=filters,
            kernel_size=1,
            strides=1,
            activation_fn=None,
            prefix=prefix+'Conv3',
            **conf_dict)

    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        x = x + shortcut
    return x



def _SeparableDepthwiseConv(inputs, filters, kernel_size=3, strides=2, padding='same', normalization=BatchNormalization, activation=ReLU, prefix=None, **conf_dict):
    x=_DeptwiseConv(inputs, kernel_size=kernel_size, strides=strides, padding=padding,
                normalization=BatchNormalization, activation=ReLU6,
                prefix=prefix, **conf_dict)
    x=_Conv(x, filters=filters, kernel_size=1, strides=1, padding='valid', 
        normalization=normalization, activation=activation,
        prefix=prefix, **conf_dict)

    return x
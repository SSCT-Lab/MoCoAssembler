import tensorflow as tf
from model.customLayer import ReLU6, HSwish6, _depth, _IBN, _Conv, backend, _Tucker, _Fused

def MobileNetV3Small(x, config=None):
    '''
        Reference:
                "Searching for MobileNetV3 (ICCV 2019)"
    '''
    alpha=config["model_config"]["backbone"]["width_multiplier"]
    conf_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["backbone"]["regularization"]),
        'kernel_initializer': tf.keras.initializers.HeUniform(),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }

    x=_Conv(x, filters=_depth(16*alpha), kernel_size=3, strides=2, activation = HSwish6, prefix='Initial/', **conf_dict)

    x=_IBN(x, expansion=1, filters=_depth(16*alpha), kernel_size=3, strides=2, activation=ReLU6, block_id=0, **conf_dict)
    out1=x

    x=_IBN(x, expansion=72./16, filters= _depth(24*alpha), kernel_size=3, strides=2, activation=ReLU6, attentionMode=None, block_id=1, **conf_dict)
    x=_IBN(x, expansion=88./24, filters= _depth(24*alpha), kernel_size=3, strides=1, activation=ReLU6, attentionMode=None, block_id=2, **conf_dict)
    out2=x

    x=_IBN(x, expansion=4, filters=_depth(40*alpha), kernel_size=5, strides=2, activation=HSwish6, block_id=3, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(40*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=4, **conf_dict) 
    x=_IBN(x, expansion=6, filters=_depth(40*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=5, **conf_dict)
    x=_IBN(x, expansion=3, filters=_depth(48*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=6, **conf_dict)
    x=_IBN(x, expansion=3, filters=_depth(48*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=7, **conf_dict)
    out3=x

    x, out3=_IBN(x, expansion=6, filters=_depth(48*alpha), kernel_size=5, strides=2, activation=HSwish6, block_id=8, Detour=True, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(48*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=9, **conf_dict) 
    x=_IBN(x, expansion=6, filters=_depth(48*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=10, **conf_dict)

    x=_Conv(x, filters=_depth(backend.int_shape(x)[-1]*6) if alpha <= 1.0 else _depth(backend.int_shape(x)[-1]*6*alpha), 
            kernel_size=1, 
            strides=1, 
            activation = HSwish6, 
            prefix='Last/', **conf_dict)
    return out1, out2, out3, x

def MobileNetV3Large(x, config=None):
    '''
        Reference:
                "Searching for MobileNetV3 (ICCV 2019)"
    '''
    alpha=config["model_config"]["backbone"]["width_multiplier"]
    conf_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["backbone"]["regularization"]),
        'kernel_initializer': tf.keras.initializers.HeUniform(),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }

    x=_Conv(x, filters=_depth(16*alpha), kernel_size=3, strides=2, activation = HSwish6, prefix='Initial/', **conf_dict)
    x=_IBN(x, expansion=1, filters=_depth(16*alpha), kernel_size=3, strides=1, activation=ReLU6, attentionMode=None, block_id=0, **conf_dict)

    x=_IBN(x, expansion=4, filters=_depth(24*alpha), kernel_size=3, strides=2, activation=ReLU6, attentionMode=None, block_id=1, **conf_dict)
    x=_IBN(x, expansion=3, filters=_depth(24*alpha), kernel_size=3, strides=1, activation=ReLU6, attentionMode=None, block_id=2, **conf_dict)
    out1=x

    x=_IBN(x, expansion=3, filters=_depth(40*alpha), kernel_size=5, strides=2, activation=ReLU6,  block_id=3, **conf_dict)
    x=_IBN(x, expansion=3, filters=_depth(40*alpha), kernel_size=5, strides=1, activation=ReLU6,  block_id=4, **conf_dict)
    x=_IBN(x, expansion=3, filters=_depth(40*alpha), kernel_size=5, strides=1, activation=ReLU6,  block_id=5, **conf_dict)
    out2=x

    x=_IBN(x, expansion=6, filters=_depth(80*alpha), kernel_size=3, strides=2, activation=HSwish6, attentionMode=None, block_id=6, **conf_dict)
    x=_IBN(x, expansion=2.5, filters=_depth(80*alpha), kernel_size=3, strides=1, activation=HSwish6, attentionMode=None, block_id=7, **conf_dict)
    x=_IBN(x, expansion=2.3, filters=_depth(80*alpha), kernel_size=3, strides=1, activation=HSwish6, attentionMode=None, block_id=8, **conf_dict)
    x=_IBN(x, expansion=2.3, filters=_depth(80*alpha), kernel_size=3, strides=1, activation=HSwish6, attentionMode=None, block_id=9, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(112*alpha), kernel_size=3, strides=1, activation=HSwish6,  block_id=10, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(112*alpha), kernel_size=3, strides=1, activation=HSwish6,  block_id=11, **conf_dict)
    out3=x

    x, out3=_IBN(x, expansion=6, filters=_depth(80*alpha), kernel_size=5, strides=2, activation=HSwish6, block_id=12, Detour=True, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(80*alpha), kernel_size=5, strides=1, activation=HSwish6,  block_id=13, **conf_dict)
    x=_IBN(x, expansion=6, filters=_depth(80*alpha), kernel_size=5, strides=1, activation=HSwish6,  block_id=14, **conf_dict)
    x=_Conv(x, filters=_depth(backend.int_shape(x)[-1]*6) if alpha <= 1.0 else _depth(backend.int_shape(x)[-1]*6*alpha), 
            kernel_size=1, 
            strides=1, 
            activation = HSwish6, 
            prefix='Last/', **conf_dict)
    return out1, out2, out3, x
import tensorflow as tf
from absl import logging
from model.customLayer import _Conv, _SeparableConv, _SeparableDepthwiseConv

def FPN(x, config=None):
    agg_mode=config["model_config"]["neck"]["aggregation"].upper()
    Uplayer = tf.keras.layers.UpSampling2D
    if agg_mode == 'ADD':
        AggLayer = tf.keras.layers.Add
        filters=config["model_config"]["neck"]["filters"]
    elif agg_mode == 'CONCAT':
        AggLayer = tf.keras.layers.Concatenate
        filters=[f//2 for f in config["model_config"]["neck"]["filters"]]
    else:
        raise ValueError("{} is not implemented yet.".format(agg_mode))
    logging.warning('FPN Aggregation: {}'.format(agg_mode))

    baseConv=_SeparableConv if config["model_config"]["neck"]["isLite"] else _Conv
    config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["neck"]["regularization"]),
        'kernel_initializer': tf.keras.initializers.HeNormal(),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }
    
    C2, C3, C4, C5=x

    P5=_Conv(C5, filters=filters[0], kernel_size=1, strides=1, prefix="C5P5/", **config_dict)

    P5_upsampled=Uplayer(name='P5_upsample')(P5)
    P4=_Conv(C4, filters=filters[0], kernel_size=1, strides=1, prefix="C4P4/", **config_dict)
    P4=AggLayer(name='P5P4_merge')([P5_upsampled, P4])

    P4=baseConv(P4, filters=filters[1], kernel_size=3, strides=1, prefix="P4_project/", **config_dict)

    P4_upsampled=Uplayer(name='P4_upsample')(P4)
    P3=_Conv(C3, filters=filters[1], kernel_size=1, strides=1, prefix="C3P3/", **config_dict)
    P3=AggLayer(name='P4P3_merge')([P4_upsampled, P3])

    P3=baseConv(P3, filters=filters[2], kernel_size=3, strides=1, prefix="P3_project/", **config_dict)

    P3_upsampled=Uplayer(name='P3_upsample')(P3)
    P2=_Conv(C2, filters=filters[2], kernel_size=1, strides=1, prefix="C2P2/", **config_dict)
    P2=AggLayer(name='C2P2_merge')([P3_upsampled, P2])
    
    P2=baseConv(P2, filters=filters[3], kernel_size=3, strides=1, prefix="P2_project/", **config_dict)
    return P2
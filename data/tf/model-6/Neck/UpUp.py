import tensorflow as tf
from functools import partial
from absl import logging
from model.customLayer import  _TransposeConv

def UPUP(x, config=None):
    C2, C3, C4, C5=x
    #C5 = tf.keras.layers.Dropout(rate=0.5)(C5)

    up_mode=config["model_config"]["neck"]["upsample"].upper()
    if up_mode == 'UPSAMPLE':
        P4=tf.keras.layers.UpSampling2D(name="C5P4")(C5)
        P3=tf.keras.layers.UpSampling2D(name="P4P3")(P4)
        P2=tf.keras.layers.UpSampling2D(name="P3P2")(P3)
    elif up_mode == 'TCONV':
        filters=config["model_config"]["neck"]["filters"]
        config_dict = {
            'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["neck"]["regularization"]),
            'kernel_initializer': tf.keras.initializers.HeNormal(),
            'trainable':not config["model_config"]["backbone"]["isFreeze"],
            'use_bias':False
        }
        P4=_TransposeConv(C5, filters=filters[0], kernel_size=4, strides=2, prefix="C5P4", **config_dict)
        P3=_TransposeConv(P4, filters=filters[1], kernel_size=4, strides=2, prefix="P4P3", **config_dict)
        P2=_TransposeConv(P3, filters=filters[2], kernel_size=4, strides=2, prefix="P3P2", **config_dict)
    else:
        raise ValueError("{} is not implemented yet.".format(up_mode))
    logging.warning('FPN UP: {}'.format(up_mode))

    return P2
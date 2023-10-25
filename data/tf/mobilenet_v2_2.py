from keras.layers import Input
from keras.models import Model
import tensorflow as tf


def conv_bn_ac(input_tensor, f_size, k_size=(3, 3), strides=1, padding='same'):
    out = tf.keras.layers.Conv2D(f_size, k_size, strides, padding=padding)(input_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.nn.relu6(out)
    return out


def bottleneck(input_tensor, f_size, strides=1, use_res=True):
    out = conv_bn_ac(input_tensor, f_size, k_size=1, padding='valid')

    out = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same',
                                          depth_multiplier=1)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.nn.relu6(out)

    out = conv_bn_ac(out, f_size, k_size=1, padding='valid')
    if use_res:
        out = tf.keras.layers.Add()([input_tensor, out])
    return out


def residual_block(input_tensor, f_size, strides, count):
    out = bottleneck(input_tensor, f_size, strides, use_res=False)
    for i in range(1, count):
        out = bottleneck(out, f_size, strides=1, use_res=True)
    return out


def net(input_shape=(224, 224, 3), alpha=1):
    input_tensor = Input(input_shape)

    if alpha not in [1, 0.75, 0.5, 0.25]:
        raise ValueError('在网络构件中，alpha的值不在[1, 0.75, 0.5, 0.25]其中！')

    f_sizes = [int(32 * alpha), int(16 * alpha), int(24 * alpha),
               int(32 * alpha), int(64 * alpha), int(96 * alpha),
               int(160 * alpha), int(320 * alpha), int(1280 * alpha)]

    bottleneck1_1 = conv_bn_ac(input_tensor, f_size=f_sizes[0], k_size=3, strides=2)

    bottleneck2_1 = residual_block(bottleneck1_1, f_size=f_sizes[1], strides=1, count=1)

    bottleneck3_1 = residual_block(bottleneck2_1, f_size=f_sizes[1], strides=2, count=6)
    bottleneck3_2 = residual_block(bottleneck3_1, f_size=f_sizes[2], strides=1, count=6)

    bottleneck4_1 = residual_block(bottleneck3_2, f_size=f_sizes[2], strides=2, count=6)
    bottleneck4_2 = residual_block(bottleneck4_1, f_size=f_sizes[2], strides=1, count=6)
    bottleneck4_3 = residual_block(bottleneck4_2, f_size=f_sizes[3], strides=1, count=6)

    bottleneck5_1 = residual_block(bottleneck4_3, f_size=f_sizes[3], strides=2, count=6)
    bottleneck5_2 = residual_block(bottleneck5_1, f_size=f_sizes[3], strides=1, count=6)
    bottleneck5_3 = residual_block(bottleneck5_2, f_size=f_sizes[3], strides=1, count=6)
    bottleneck5_4 = residual_block(bottleneck5_3, f_size=f_sizes[4], strides=1, count=6)

    bottleneck6_1 = residual_block(bottleneck5_4, f_size=f_sizes[4], strides=1, count=6)
    bottleneck6_2 = residual_block(bottleneck6_1, f_size=f_sizes[4], strides=1, count=6)
    bottleneck6_3 = residual_block(bottleneck6_2, f_size=f_sizes[5], strides=1, count=6)

    bottleneck7_1 = residual_block(bottleneck6_3, f_size=f_sizes[5], strides=2, count=6)
    bottleneck7_2 = residual_block(bottleneck7_1, f_size=f_sizes[5], strides=1, count=6)
    bottleneck7_3 = residual_block(bottleneck7_2, f_size=f_sizes[6], strides=1, count=6)

    bottleneck8_1 = residual_block(bottleneck7_3, f_size=f_sizes[7], strides=1, count=6)

    bottleneck9_1 = conv_bn_ac(bottleneck8_1, f_size=f_sizes[8], k_size=1, strides=1, padding='valid')

    pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(bottleneck9_1)

    bottleneck10_1 = conv_bn_ac(pool, f_size=1000, k_size=1, strides=1, padding='valid')

    model = Model(input_tensor, bottleneck10_1)
    return model


def mobilenet_v2_2():
    return net()

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def conv_bn_ac(input_tensor, f_size, k_size=(3, 3), strides=1, padding='same'):
    out = tf.keras.layers.Conv2D(f_size, k_size, strides, padding=padding)(input_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.nn.relu6(out)
    return out


def dw_pw_conv(input_tensor, f_size, strides=1):
    out = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same',
                          depth_multiplier=1)(input_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.nn.relu6(out)
    # （1*1卷积）
    out = conv_bn_ac(out, f_size, k_size=1, padding='valid')
    return out


def net(input_shape=(224, 224, 3), alpha=1):
    input_tensor = Input(input_shape)

    if alpha not in [1, 0.75, 0.5, 0.25]:
        raise ValueError('在网络构件中，alpha的值不在[1, 0.75, 0.5, 0.25]其中！')

    f_sizes = [int(32 * alpha), int(64 * alpha), int(128 * alpha),
               int(256 * alpha), int(512 * alpha), int(1024 * alpha)]  # 滤波器大小设置
    conv1 = conv_bn_ac(input_tensor, f_size=f_sizes[0], strides=2)

    conv2 = dw_pw_conv(conv1, f_size=f_sizes[1])

    conv3 = dw_pw_conv(conv2, f_size=f_sizes[2], strides=2)

    conv4 = dw_pw_conv(conv3, f_size=f_sizes[2])
    conv4 = dw_pw_conv(conv4, f_size=f_sizes[3], strides=2)

    conv5 = dw_pw_conv(conv4, f_size=f_sizes[3])
    conv5 = dw_pw_conv(conv5, f_size=f_sizes[4], strides=2)

    conv6 = dw_pw_conv(conv5, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])

    conv6 = dw_pw_conv(conv6, f_size=f_sizes[5], strides=2)

    conv7 = dw_pw_conv(conv6, f_size=f_sizes[5])
    pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(conv7)
    print(pool.shape)

    out = tf.keras.layers.Dense(1000, activation='softmax')(pool)

    model = Model(input_tensor, out)
    return model


if __name__ == '__main__':
    model = net()
    model.summary()

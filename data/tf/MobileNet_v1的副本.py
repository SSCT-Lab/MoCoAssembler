from tensorflow.keras.layers import Conv2D, DepthwiseConv2D,AveragePooling2D, \
    Input, BatchNormalization, Activation, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_bn_ac(input_tensor, f_size, k_size=(3,3), strides=1, padding='same'):
    '''
    :param input_tensor:    输入的tensor
    :param f_size:          filters size
    :param k_size:          kernel size
    :param strides:         步长
    :param padding:         填充方式
    :return:
    '''
    # 普通卷积+BN+激活
    out = Conv2D(f_size, k_size, strides, padding=padding)(input_tensor)
    out = BatchNormalization()(out)
    out = tf.nn.relu6(out)
    return out


def dw_pw_conv(input_tensor, f_size, strides=1):
    '''
    :param input_tensor:    输入的tensor
    :param f_size:          filters size
    :param strides:         步长
    :return:
    '''
    # 深度可分离卷积+逐点卷积（1*1卷积）
    out = DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same',
                          depth_multiplier=1)(input_tensor)
    out = BatchNormalization()(out)
    out = tf.nn.relu6(out)
    # （1*1卷积）
    out = conv_bn_ac(out, f_size, k_size=1, padding='valid')
    return out


def net(input_shape=(224, 224, 3), alpha=1):
    input_tensor = Input(input_shape)

    if alpha not in [1, 0.75, 0.5, 0.25]:
        raise ValueError('在网络构件中，alpha的值不在[1, 0.75, 0.5, 0.25]其中！')

    f_sizes = [int(32*alpha), int(64*alpha), int(128*alpha),
               int(256*alpha), int(512*alpha), int(1024*alpha)] # 滤波器大小设置
    # 输入以 224，224，3 为例
    # 224,224,3 -> 112,112,32
    conv1 = conv_bn_ac(input_tensor, f_size=f_sizes[0], strides=2)

    # 112,112,32 -> 112,112,64
    conv2 = dw_pw_conv(conv1, f_size=f_sizes[1])

    # 112,112,64 -> 56,56,128
    conv3 = dw_pw_conv(conv2, f_size=f_sizes[2], strides=2)

    # 56,56,128 -> 56,56,128 -> 28,28,256
    conv4 = dw_pw_conv(conv3, f_size=f_sizes[2])
    conv4 = dw_pw_conv(conv4, f_size=f_sizes[3], strides=2)

    # 28,28,256 -> 28,28,256 -> 14,14,512
    conv5 = dw_pw_conv(conv4, f_size=f_sizes[3])
    conv5 = dw_pw_conv(conv5, f_size=f_sizes[4], strides=2)

    # 14,14,512 -> 14,14,512(5次) -> 7,7,1024
    conv6 = dw_pw_conv(conv5, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])
    conv6 = dw_pw_conv(conv6, f_size=f_sizes[4])

    conv6 = dw_pw_conv(conv6, f_size=f_sizes[5], strides=2)

    # 7,7,1024 -> 7,7,1024
    conv7 = dw_pw_conv(conv6, f_size=f_sizes[5])
    pool = AveragePooling2D(pool_size=(7,7))(conv7)
    print(pool.shape)

    # 7,7,1024 -> 7,7,1000
    out = Dense(1000, activation='softmax')(pool)

    model = Model(input_tensor, out)
    return model


if __name__ == '__main__':
    model = net()
    model.summary()


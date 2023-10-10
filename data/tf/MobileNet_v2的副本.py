from tensorflow.keras.layers import Conv2D, DepthwiseConv2D,AveragePooling2D, \
    Input, BatchNormalization, Activation, Dense, Add
from tensorflow.keras.models import Model
import tensorflow as tf

'''
# relu6的使用方法
import tensorflow.keras.backend as K
def relu6():
    return K.relu(max_value=6)
Activation(relu6)
'''

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


def bottleneck(input_tensor, f_size, strides=1, use_res=True):
    '''
    :param input_tensor:    输入的tensor
    :param f_size:          filters size
    :param strides:         步长
    :param use_res:         是否使用残差连接
    :return:
    '''
    # 逐点卷积+深度可分离卷积+逐点卷积（1*1卷积）
    out = conv_bn_ac(input_tensor, f_size, k_size=1, padding='valid')

    out = DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same',
                          depth_multiplier=1)(out)
    out = BatchNormalization()(out)
    out = tf.nn.relu6(out)
    # （1*1卷积）
    out = conv_bn_ac(out, f_size, k_size=1, padding='valid')
    if use_res:
        out = Add()([input_tensor, out])
    return out

def residual_block(input_tensor, f_size, strides, count):
    '''
    :param input_tensor:    输入的tensor
    :param f_size:          filters size
    :param strides:         步长
    :param count:           循环次数
    :return:
    '''
    out = bottleneck(input_tensor, f_size, strides, use_res=False)
    for i in range(1, count):
        out = bottleneck(out, f_size, strides=1, use_res=True)
    return out


def net(input_shape=(224, 224, 3), alpha=1):
    input_tensor = Input(input_shape)

    if alpha not in [1, 0.75, 0.5, 0.25]:
        raise ValueError('在网络构件中，alpha的值不在[1, 0.75, 0.5, 0.25]其中！')

    f_sizes = [int(32*alpha), int(16*alpha), int(24*alpha),
               int(32*alpha), int(64*alpha), int(96*alpha),
               int(160*alpha), int(320*alpha), int(1280*alpha)] # 滤波器大小设置, 通过alpha方便调节滤波器数量

    # 224,224,3 -> 224,224,32
    bottleneck1_1 = conv_bn_ac(input_tensor, f_size=f_sizes[0], k_size=3, strides=2)

    # 224,224,32 -> 112,112,16
    bottleneck2_1 = residual_block(bottleneck1_1, f_size=f_sizes[1], strides=1, count=1)

    # 112,112,16 -> 112,112,16 -> 56,56,24
    bottleneck3_1 = residual_block(bottleneck2_1, f_size=f_sizes[1], strides=2, count=6)
    bottleneck3_2 = residual_block(bottleneck3_1, f_size=f_sizes[2], strides=1, count=6)

    # 56,56,24 -> 28,28,24 -> 28,28,24 -> 28,28,32
    bottleneck4_1 = residual_block(bottleneck3_2, f_size=f_sizes[2], strides=2, count=6)
    bottleneck4_2 = residual_block(bottleneck4_1, f_size=f_sizes[2], strides=1, count=6)
    bottleneck4_3 = residual_block(bottleneck4_2, f_size=f_sizes[3], strides=1, count=6)

    # 28,28,32 -> 14,14,32 -> 14,14,32 -> 14,14,32 -> 14,14,64
    bottleneck5_1 = residual_block(bottleneck4_3, f_size=f_sizes[3], strides=2, count=6)
    bottleneck5_2 = residual_block(bottleneck5_1, f_size=f_sizes[3], strides=1, count=6)
    bottleneck5_3 = residual_block(bottleneck5_2, f_size=f_sizes[3], strides=1, count=6)
    bottleneck5_4 = residual_block(bottleneck5_3, f_size=f_sizes[4], strides=1, count=6)

    # 14,14,64 -> 14,14,64 -> 14,14,96
    bottleneck6_1 = residual_block(bottleneck5_4, f_size=f_sizes[4], strides=1, count=6)
    bottleneck6_2 = residual_block(bottleneck6_1, f_size=f_sizes[4], strides=1, count=6)
    bottleneck6_3 = residual_block(bottleneck6_2, f_size=f_sizes[5], strides=1, count=6)

    # 14,14,96 -> 7,7,96 -> 7,7,160
    bottleneck7_1 = residual_block(bottleneck6_3, f_size=f_sizes[5], strides=2, count=6)
    bottleneck7_2 = residual_block(bottleneck7_1, f_size=f_sizes[5], strides=1, count=6)
    bottleneck7_3 = residual_block(bottleneck7_2, f_size=f_sizes[6], strides=1, count=6)

    # 7,7,160 -> 7,7,320
    bottleneck8_1 = residual_block(bottleneck7_3, f_size=f_sizes[7], strides=1, count=6)

    # 7,7,320 -> 7,7,1280
    bottleneck9_1 = conv_bn_ac(bottleneck8_1, f_size=f_sizes[8], k_size=1, strides=1, padding='valid')

    # 7,7,1280 -> 1,1,1280
    pool = AveragePooling2D(pool_size=(7,7))(bottleneck9_1)

    # 1,1,1280 -> 1,1,1000
    bottleneck10_1 = conv_bn_ac(pool, f_size=1000, k_size=1, strides=1, padding='valid')

    model = Model(input_tensor, bottleneck10_1)
    return model


if __name__ == '__main__':
    model = net()
    model.summary()
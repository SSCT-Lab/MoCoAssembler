import tensorflow as tf


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel,
                               padding='same',
                               use_bias=False,
                               strides=strides,
                               name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.Activation(relu6, name='conv1_relu')(x)
    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False,
                                        name='conv_dw_%d' % block_id)(inputs)

    x = tf.keras.layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = tf.keras.layers.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                               padding='same',
                               use_bias=False,
                               strides=(1, 1),
                               name='conv_pw_%d' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = tf.keras.layers.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    return x


def relu6(x):
    return tf.keras.backend.relu(x, max_value=6)


def MobileNet(depth_multiplier=1):
    input_tensor = tf.keras.Input(shape=(640, 640, 3))
    # 640,640,3 -> 320,320,8
    x = _conv_block(input_tensor, filters=8, strides=(2, 2))
    # 320,320,8 -> 320,320,16
    x = _depthwise_conv_block(x, 16, depth_multiplier, block_id=1)

    # 320,320,16 -> 160,160,32
    x = _depthwise_conv_block(x, 32, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 32, depth_multiplier, block_id=3)

    # 160,160,32 -> 80,80,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=5)

    # 80,80,64 -> 40,40,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=11)

    # 40,40,128 -> 20,20,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=13)
    output_tensor = x
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def mobilenet025():
    return MobileNet()

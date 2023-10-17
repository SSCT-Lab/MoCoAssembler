import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def densenet_model(growth_rate=32, nb_filter=64, nb_layers=[6, 12, 24, 16], reduction=0.0,
                   dropout_rate=0.0, classes=16, shape=(32, 32, 3), batch_size=32,
                   with_output_block=True, with_se_layers=True):
    compression = 1.0 - reduction

    nb_dense_block = len(nb_layers)
    img_input = Input(shape=shape, name='data')

    x = tf.keras.layers.ZeroPadding2D((3, 3), name='conv1_zeropadding', batch_size=batch_size)(img_input)
    x = tf.keras.layers.Conv2D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    stage = 0
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate)

        if with_se_layers:
            x = se_block(x, stage, 'dense', nb_filter)

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

        if with_se_layers:
            x = se_block(x, stage, 'transition', nb_filter)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate)

    if with_se_layers:
        x = se_block(x, final_stage, 'dense', nb_filter)

    x = tf.keras.layers.BatchNormalization(name='conv_final_blk_bn')(x)
    x = tf.keras.layers.Activation('relu', name='relu_final_blk')(x)

    if not with_output_block:
        return Model(inputs=img_input, outputs=x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='pool_final')(x)
    x = tf.keras.layers.Dense(classes, name='fc6')(x)
    output = tf.keras.layers.Activation('softmax', name='prob')(x)

    return Model(inputs=img_input, outputs=output)


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    inter_channel = nb_filter * 4
    x = tf.keras.layers.BatchNormalization(name=conv_name_base + '_x1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base + '_x1')(x)
    x = tf.keras.layers.Conv2D(inter_channel, 1, 1, name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.BatchNormalization(name=conv_name_base + '_x2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base + '_x2')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = tf.keras.layers.Conv2D(nb_filter, 3, 1, name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def se_block(x, stage, previous, nb_filter, ratio=16):
    se_name = 'se' + str(stage) + '_' + previous
    init = x
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_2d_' + se_name)(x)
    x = tf.keras.layers.Dense(nb_filter // ratio, name='dense_relu_' + se_name)(x)
    x = tf.keras.layers.Activation('relu', name='relu_' + se_name)(x)
    x = tf.keras.layers.Dense(nb_filter, name='dense_sigmoid_' + se_name)(x)
    x = tf.keras.layers.Activation('sigmoid', name='sigmoid_' + se_name)(x)
    x = tf.expand_dims(x, 1)
    x = init * tf.expand_dims(x, 1)
    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                grow_nb_filters=True):
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = tf.concat([concat_feat, x], -1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = tf.keras.layers.BatchNormalization(name=conv_name_base + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=relu_name_base)(x)
    x = tf.keras.layers.Conv2D(int(nb_filter * compression), 1, 1, name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x

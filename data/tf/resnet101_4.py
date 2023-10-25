import tensorflow as tf


def normalize(axis=-1, momentum=0.9, epsilon=1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, **kwargs)


def res_basic_block(x, n_feature, stride_size=1, normalize=normalize, activation=tf.keras.activations.relu, prefix="",
                    **kwargs):
    out = tf.keras.layers.Conv2D(n_feature, kernel_size=3, strides=stride_size, padding="SAME", use_bias=False,
                                 kernel_initializer="he_normal", name=prefix + "conv1")(x)
    out = normalize(name=prefix + "norm1")(out)
    out = tf.keras.layers.Activation(activation, name=prefix + "act1")(out)
    out = tf.keras.layers.Conv2D(n_feature, kernel_size=3, padding="SAME", use_bias=False,
                                 kernel_initializer="he_normal", name=prefix + "conv2")(out)
    out = normalize(name=prefix + "norm2")(out)
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != n_feature:  # downsample
        x = tf.keras.layers.Conv2D(n_feature, kernel_size=1, strides=stride_size, padding="SAME", use_bias=False,
                                   kernel_initializer="he_normal", name=prefix + "downsample_conv")(x)
        x = normalize(name=prefix + "downsample_norm")(x)
    out = tf.keras.layers.Add(name=prefix + "residual")([out, x])
    out = tf.keras.layers.Activation(activation, name=prefix + "out")(out)
    return out


def res_bottleneck_block(x, n_feature, stride_size=1, group_size=1, base_width=64, expansion=4, normalize=normalize,
                         activation=tf.keras.activations.relu, prefix="", **kwargs):
    width = int(n_feature * (base_width / 64.)) * group_size
    out = tf.keras.layers.Conv2D(width, kernel_size=1, strides=1, padding="SAME", use_bias=False,
                                 kernel_initializer="he_normal", name=prefix + "conv1")(x)
    out = normalize(name=prefix + "norm1")(out)
    out = tf.keras.layers.Activation(activation, name=prefix + "act1")(out)
    out = tf.keras.layers.Conv2D(width, kernel_size=3, strides=stride_size, groups=group_size, padding="SAME",
                                 use_bias=False, kernel_initializer="he_normal", name=prefix + "conv2")(out)
    out = normalize(name=prefix + "norm2")(out)
    out = tf.keras.layers.Activation(activation, name=prefix + "act2")(out)
    out = tf.keras.layers.Conv2D(n_feature * expansion, kernel_size=1, strides=1, padding="SAME", use_bias=False,
                                 kernel_initializer="he_normal", name=prefix + "conv3")(out)
    out = normalize(name=prefix + "norm3")(out)
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != n_feature * expansion:  # downsample
        x = tf.keras.layers.Conv2D(n_feature * expansion, kernel_size=1, strides=stride_size, padding="SAME",
                                   use_bias=False, kernel_initializer="he_normal", name=prefix + "downsample_conv")(x)
        x = normalize(name=prefix + "downsample_norm")(x)
    out = tf.keras.layers.Add(name=prefix + "residual")([out, x])
    out = tf.keras.layers.Activation(activation, name=prefix + "out")(out)
    return out


def res_stack(x, n_block, n_feature, stride_size=1, group_size=1, base_width=64, block=res_bottleneck_block,
              normalize=normalize, activation=tf.keras.activations.relu, prefix=""):
    out = block(x, n_feature=n_feature, stride_size=stride_size, group_size=group_size, base_width=base_width,
                normalize=normalize, activation=activation, prefix=prefix + "block1_")
    for index in range(1, n_block):
        out = block(out, n_feature=n_feature, group_size=group_size, base_width=base_width, normalize=normalize,
                    activation=activation, prefix=prefix + "block{0}_".format(index + 1))
    return out


def resnet(x, n_blocks, block, n_class=1000, include_top=False, group_size=1, base_width=64, normalize=normalize,
           activation=tf.keras.activations.relu):
    # stem
    out = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", use_bias=False,
                                 kernel_initializer="he_normal", name="stem_conv")(x)
    out = normalize(name="stem_norm")(out)
    out = tf.keras.layers.Activation(activation, name="stem_act")(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="SAME", name="stem_pooling")(out)

    # stage
    out = res_stack(out, n_blocks[0], 64, group_size=group_size, base_width=base_width, block=block,
                    normalize=normalize, activation=activation, prefix="stage1_")
    out = res_stack(out, n_blocks[1], 128, stride_size=2, group_size=group_size, base_width=base_width, block=block,
                    normalize=normalize, activation=activation, prefix="stage2_")
    out = res_stack(out, n_blocks[2], 256, stride_size=2, group_size=group_size, base_width=base_width, block=block,
                    normalize=normalize, activation=activation, prefix="stage3_")
    out = res_stack(out, n_blocks[3], 512, stride_size=2, group_size=group_size, base_width=base_width, block=block,
                    normalize=normalize, activation=activation, prefix="stage4_")

    # fc
    if include_top:
        out = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(out)
        out = tf.keras.layers.Dense(n_class, use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros",
                                    name="logits")(out)
    return out


def resnet101_4(input_tensor=None, input_shape=None, classes=1000, include_top=True, weights="imagenet"):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    out = resnet(img_input, [3, 4, 23, 3], res_bottleneck_block, classes, include_top, normalize=normalize,
                 activation=tf.keras.activations.relu)
    model = tf.keras.Model(img_input, out)

    return model

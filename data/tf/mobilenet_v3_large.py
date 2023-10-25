import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = h_sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = inputs * branch
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, in_size, exp_size, out_size, s, is_se_existing, NL, k):
        super(BottleNeck, self).__init__()
        self.stride = s
        self.in_size = in_size
        self.out_size = out_size
        self.is_se_existing = is_se_existing
        self.NL = NL
        self.conv1 = tf.keras.layers.Conv2D(filters=exp_size,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=s,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=exp_size)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_size,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.NL == "HS":
            x = h_swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        if self.NL == "HS":
            x = h_swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)
        if self.is_se_existing:
            x = self.se(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = tf.keras.layers.add([x, inputs])

        return x


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)

class MobileNetV3Large(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)
        self.bneck8 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck9 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck10 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)
        self.bneck11 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
        self.bneck12 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)
        self.bneck13 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck14 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck15 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)

        self.conv2 = tf.keras.layers.Conv2D(filters=960,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                                        strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=10,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)
        x = self.bneck12(x, training=training)
        x = self.bneck13(x, training=training)
        x = self.bneck14(x, training=training)
        x = self.bneck15(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = h_swish(x)
        x = self.conv4(x)

        return x


def mobilenet_v3_large():
    return MobileNetV3Large()

import tensorflow as tf
from tensorflow.keras import models, Sequential


def GoogLeNet(im_height=224, im_width=224, class_num=1000, aux_logits=False):
    input_image = tf.keras.layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)

    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)

    if aux_logits:
        aux1 = InceptionAux(class_num, name="aux_1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)

    if aux_logits:
        aux2 = InceptionAux(class_num, name="aux_2")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)

    x = tf.keras.layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)

    x = tf.keras.layers.Flatten(name="output_flatten")(x)
    x = tf.keras.layers.Dropout(rate=0.4, name="output_dropout")(x)
    x = tf.keras.layers.Dense(class_num, name="output_dense")(x)
    aux3 = tf.keras.layers.Softmax(name="aux_3")(x)

    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model


class Inception(tf.keras.layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):  
        super(Inception, self).__init__(**kwargs)

        self.branch1 = tf.keras.layers.Conv2D(ch1x1, kernel_size=1, activation="relu")

        self.branch2 = Sequential([  
            tf.keras.layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])

        self.branch3 = Sequential([
            tf.keras.layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])

        self.branch4 = Sequential([
            tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),
            tf.keras.layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = tf.keras.layers.AvgPool2D(pool_size=5, strides=3)  # 平均池化
        self.conv = tf.keras.layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.averagePool(inputs)
        x = self.conv(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = self.fc1(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



import tensorflow as tf
from tensorflow.python.keras import Input, layers, models


def GoogLeNet(class_num=1000, input_shape=(224, 224, 3), aux_logits=True):
    input_tensor = Input(shape=input_shape, dtype="float32")

    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu", name="layers.conv2d_1")(input_tensor)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="layers.maxpool_1")(x)

    x = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu", name="layers.conv2d_2")(x)
    x = layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu", name="layers.conv2d_3")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="layers.maxpool_2")(x)

    x = inception(x, 64, 96, 128, 16, 32, 32)
    x = inception(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="layers.maxpool_3")(x)

    x = inception(x, 192, 96, 208, 16, 48, 64)
    if aux_logits:
        aux1 = inceptionAux(x, class_num)

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)
    if aux_logits:
        aux2 = inceptionAux(x, class_num)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="layers.maxpool_4")(x)

    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = inception(x, 384, 192, 384, 48, 128, 128)
    x = layers.AveragePooling2D(pool_size=7, strides=1, name="avgpool_1")(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(rate=0.4)(x)
    x = layers.Dense(class_num, activation="softmax", name="output")(x)
    output_tensor = x

    if aux_logits:
        model = models.Model(inputs=input_tensor, outputs=[aux1, aux2, output_tensor])
    else:
        model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
    x1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")(inputs)

    x2 = layers.Conv2D(ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x2 = layers.Conv2D(ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")(x2)

    x3 = layers.Conv2D(ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x3 = layers.Conv2D(ch5x5, kernel_size=5, strides=1, padding="same", activation="relu")(x3)

    x4 = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(inputs)
    x4 = layers.Conv2D(pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")(x4)

    outputs = layers.concatenate([x1, x2, x3, x4])
    return outputs


def inceptionAux(inputs, class_num):

    x = layers.AveragePooling2D(pool_size=5, strides=3)(inputs)
    x = layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(rate=0.7)(x)
    x = layers.Dense(1024, activation="relu")(x)

    x = layers.Dropout(rate=0.7)(x)
    x = layers.Dense(class_num, activation="softmax")(x)

    x = layers.concatenate([x])
    return x


if __name__ == '__main__':
    net = GoogLeNet(class_num=3, input_shape=(224, 224, 3))
    net.summary()

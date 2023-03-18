import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model


def SqueezeNet(input_shape):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding="same")(input_tensor)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # 2nd block  x = FireModule(s1=16, e1=64, e3=64)(x)
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)

    # 3rd block  x = FireModule(s1=16, e1=64, e3=64)(x)
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)

    # 4th block  x = FireModule(s1=32, e1=128, e3=128)(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # 5th block  x = FireModule(s1=32, e1=128, e3=128)(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)

    # 6th block  x = FireModule(s1=48, e1=192, e3=192)(x)
    x = Conv2D(filters=48, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)

    # 7th block x = FireModule(s1=48, e1=192, e3=192)(x)
    x = Conv2D(filters=48, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)

    # 8th block  x = FireModule(s1=64, e1=256, e3=256)(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # 9th block  x = FireModule(s1=64, e1=256, e3=256)(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.nn.relu(x)
    y1 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")(x)
    y1 = tf.nn.relu(y1)
    y2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(x)
    y2 = tf.nn.relu(y2)
    x = tf.concat(values=[y1, y2], axis=-1)
    x = Dropout(rate=0.5)(x)

    # 10th block
    x = Conv2D(filters=5, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = GlobalAveragePooling2D()(x)

    # output
    output_tensor = tf.nn.softmax(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


if __name__ == '__main__':
    model = SqueezeNet((224, 224, 3))
    model.summary()
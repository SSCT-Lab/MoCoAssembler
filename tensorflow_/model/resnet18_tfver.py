import tensorflow as tf
from tensorflow.python.keras import Input, layers, models
from tensorflow.python.layers.normalization import BatchNormalization


def ResNet18(input_shape, class_nums):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation="relu")(input_tensor)
    x = BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    # 2nd block
    x = basic_bottle(x, filters=64, kernel_size=3, strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=3, strides=1, padding='same', if_baisc=False)

    # 3rd block
    x = basic_bottle(x, filters=128, kernel_size=3, strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=3, strides=1, padding='same', if_baisc=False)

    # 4th block
    x = basic_bottle(x, filters=256, kernel_size=3, strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=3, strides=1, padding='same', if_baisc=False)

    # 5th block
    x = basic_bottle(x, filters=512, kernel_size=3, strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=512, kernel_size=3, strides=1, padding='same', if_baisc=False)

    # 6th block
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(class_nums, activation='softmax')(x)
    output_tensor = x

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def basic_bottle(inpt, filters=64, kernel_size=3, strides=1, padding='same', if_baisc=False):

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(inpt)
    x = BatchNormalization()(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu")(x)
    x = BatchNormalization()(x)

    if if_baisc==True:
        temp = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', activation="relu")(inpt)
        temp = BatchNormalization()(temp)
        output = layers.add([x, temp])
    else:
        output = layers.add([x, inpt])
    return output


if __name__ == '__main__':
    model = ResNet18([28, 28, 1], 3)
    model.summary()



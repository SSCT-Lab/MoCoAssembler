import tensorflow as tf
from tensorflow import keras


def resnet50(classes=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.ZeroPadding2D((3, 3))(input_tensor)

    # 2nd block
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 3rd block
    x = inceptionA(x, 3, 64, 64, 256, strides=(1, 1))
    x = inceptionB(x, 3, 64, 64, 256)
    x = inceptionB(x, 3, 64, 64, 256)

    # 4th block
    x = inceptionA(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)

    # 5th block
    x = inceptionA(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)

    # 6th block
    x = inceptionA(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)

    # output
    x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=classes, activation='softmax')(x)
    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, kernel_size, filters1, filters2, filters3, strides=(2, 2)):
    x = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, activation='relu')(inputs)
    x = keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides)(inputs)

    shape = tf.shape(x)
    shortcut = tf.reshape(shortcut, shape)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    outputs = x
    return outputs


def inceptionB(inputs, kernel_size, filters1, filters2, filters3):
    x = keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), activation='relu')(inputs)
    x = keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)

    shape = tf.shape(x)
    inputs = tf.reshape(inputs, shape)

    x = keras.layers.add([x, inputs])
    x = keras.layers.Activation('relu')(x)

    outputs = x
    return outputs


if __name__ == '__main__':
    model = resnet50()

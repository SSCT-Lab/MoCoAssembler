import tensorflow as tf


def ResNet50(class_num=1000, input_shape=(224, 224, 3)):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_tensor)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = inceptionA(x, 3, 64, 64, 256, strides=(1, 1))
    x = inceptionB(x, 3, 64, 64, 256)
    x = inceptionB(x, 3, 64, 64, 256)

    x = inceptionA(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)
    x = inceptionB(x, 3, 128, 128, 512)

    x = inceptionA(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)
    x = inceptionB(x, 3, 256, 256, 1024)

    x = inceptionA(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)
    x = inceptionB(x, 3, 512, 512, 2048)

    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    output_tensor = tf.keras.layers.Dense(units=class_num, activation="softmax")(tf.keras.layers.Flatten()(x))

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inceptionA(inputs, kernel_size, filters1, filters2, filters3, strides=(2, 2)):
    x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)
    temp = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides)(inputs)

    outputs = tf.keras.layers.add([x, temp])
    return outputs


def inceptionB(inputs, kernel_size, filters1, filters2, filters3):
    x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1))(x)

    outputs = tf.keras.layers.add([x, inputs])
    return outputs


def resnet50_4():
    return ResNet50()

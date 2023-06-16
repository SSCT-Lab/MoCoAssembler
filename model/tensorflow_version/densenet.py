import tensorflow as tf
from tensorflow import keras


def densenet(input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape)
    features_list = []

    # 1st block
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # 2nd block
    for i in range(6):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 3rd block
    for i in range(12):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 4th block
    for i in range(24):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 5th block
    for i in range(16):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = densenet((224, 224, 3))
    model.summary()

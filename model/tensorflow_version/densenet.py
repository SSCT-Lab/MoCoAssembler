import tensorflow as tf
from tensorflow.python.keras import Input, layers, models
from tensorflow.python.layers.normalization import BatchNormalization


def DenseNet(input_shape):
    input_tensor = Input(shape=input_shape)
    features_list = []

    # 1st block
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu")(input_tensor)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # 2nd block
    for i in range(6):
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 3rd block
    for i in range(12):
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()

    x = BatchNormalization()(x)
    x = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 4th block
    for i in range(24):
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = BatchNormalization()(x)
    x = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    # 5th block
    for i in range(16):
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        features_list.append(x)
        x = tf.concat(features_list, axis=-1)
    features_list.clear()
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    output_tensor = x

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = DenseNet((224, 224, 3))
    model.summary()

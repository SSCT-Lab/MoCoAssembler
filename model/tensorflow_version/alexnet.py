import tensorflow as tf
from tensorflow import keras


def AlexNet(class_num, input_shape):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    # 1st block
    x = keras.layers.Conv2D(filters=64, kernel_size=(11, 11), activation='relu', strides=(4, 4), padding='same')(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 2nd block
    x = keras.layers.Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 3rd block
    x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # 4th block
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # 5th block
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 6th block
    x = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)

    # 7th block
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation='relu')(x)

    # 8th block
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation='relu')(x)

    # output
    x = keras.layers.Dense(units=class_num)(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = AlexNet(1000, (224, 224, 3))

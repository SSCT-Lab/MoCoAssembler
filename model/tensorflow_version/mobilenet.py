import tensorflow as tf
from tensorflow import keras


def MoblieNet(input_shape):
    input_tensor = keras.Input(shape=input_shape)

    # 1st block
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(input_tensor)

    # 2nd block
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', groups=32)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 3rd block
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=2, padding='same', groups=64)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 4th block
    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', groups=128)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='valid',)(x)

    # 5th block
    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=2, padding='same', groups=128)(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 6th block
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', groups=256)(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 7th block
    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=2, padding='same', groups=256)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 8th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 9th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 10th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 11th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 12th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 13th block
    x = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=2, padding='same', groups=512)(x)
    x = keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 14th block
    x = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', strides=2, padding='same', groups=1024)(x)
    x = keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', padding='valid')(x)

    # 15th block
    x = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=1000)(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = MoblieNet((224, 224, 3))

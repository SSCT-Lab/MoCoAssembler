import tensorflow as tf
from tensorflow import keras


def AlexNet(class_num, input_shape):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    # 1st block
    x = keras.layers.Conv2D(64, (11, 11), activation='relu', strides=(4, 4), padding='same', name='conv1')(input_tensor)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # 2nd block
    x = keras.layers.Conv2D(192, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)

    # 3rd block
    x = keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')(x)

    # 4th block
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)

    # 5th block
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)

    # 6th block
    x = keras.layers.AveragePooling2D((1, 1), strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)

    # 7th block
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation='relu', name='linear1')(x)

    # 8th block
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation='relu', name='linear2')(x)

    # output
    x = keras.layers.Dense(class_num, name='linear3')(x)

    output_tensor = x

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = AlexNet(1000, (224, 224, 3))
    model.summary()

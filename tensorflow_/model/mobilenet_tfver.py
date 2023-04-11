import tensorflow as tf
from tensorflow.python.keras import Input, layers, models


def MoblieNet(input_shape):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', name='conv0')(input_tensor)

    # 2nd block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', groups=32, name='conv1a')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='valid', name='conv1b')(x)

    # 3rd block
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same', groups=64, name='conv2a')(x)
    x = layers.Conv2D(128, (1, 1), activation='relu', padding='valid', name='conv2b')(x)

    # 4th block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', groups=128, name='conv3a')(x)
    x = layers.Conv2D(128, (1, 1), activation='relu', padding='valid', name='conv3b')(x)

    # 5th block
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2, padding='same', groups=128, name='conv4a')(x)
    x = layers.Conv2D(256, (1, 1), activation='relu', padding='valid', name='conv4b')(x)

    # 6th block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', groups=256, name='conv5a')(x)
    x = layers.Conv2D(256, (1, 1), activation='relu', padding='valid', name='conv5b')(x)

    # 7th block
    x = layers.Conv2D(256, (3, 3), activation='relu', strides=2, padding='same', groups=256, name='conv6a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv6b')(x)

    # 8th block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv7a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv7b')(x)

    # 9th block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv8a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv8b')(x)

    # 10th block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv9a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv9b')(x)

    # 11th block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv10a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv10b')(x)

    # 12th block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv11a')(x)
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv11b')(x)

    # 13th block
    x = layers.Conv2D(512, (3, 3), activation='relu', strides=2, padding='same', groups=512, name='conv12a')(x)
    x = layers.Conv2D(1024, (1, 1), activation='relu', padding='valid', name='conv12b')(x)

    # 14th block
    x = layers.Conv2D(1024, (3, 3), activation='relu', strides=2, padding='same', groups=1024, name='conv13a')(x)
    x = layers.Conv2D(1024, (1, 1), activation='relu', padding='valid', name='conv13b')(x)

    # 15th block
    x = layers.AveragePooling2D((1, 1), strides=(1, 1))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(1000, name='fc')(x)

    output_tensor = x
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = MoblieNet((224, 224, 3))
    model.summary()

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Flatten, AveragePooling2D, Dense
from tensorflow.python.keras.models import Model


def MoblieNet(input_shape):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = Conv2D(32, (3, 3), strides=2, padding='same', name='conv0')(input_tensor)

    # 2nd block
    x = Conv2D(32, (3, 3), activation='relu', padding='same', groups=32, name='conv1a')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='valid', name='conv1b')(x)

    # 3rd block
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same', groups=64, name='conv2a')(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='valid', name='conv2b')(x)

    # 4th block
    x = Conv2D(128, (3, 3), activation='relu', padding='same', groups=128, name='conv3a')(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='valid', name='conv3b')(x)

    # 5th block
    x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same', groups=128, name='conv4a')(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='valid', name='conv4b')(x)

    # 6th block
    x = Conv2D(256, (3, 3), activation='relu', padding='same', groups=256, name='conv5a')(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='valid', name='conv5b')(x)

    # 7th block
    x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same', groups=256, name='conv6a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv6b')(x)

    # 8th block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv7a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv7b')(x)

    # 9th block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv8a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv8b')(x)

    # 10th block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv9a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv9b')(x)

    # 11th block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv10a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv10b')(x)

    # 12th block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', groups=512, name='conv11a')(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='valid', name='conv11b')(x)

    # 13th block
    x = Conv2D(512, (3, 3), activation='relu', strides=2, padding='same', groups=512, name='conv12a')(x)
    x = Conv2D(1024, (1, 1), activation='relu', padding='valid', name='conv12b')(x)

    # 14th block
    x = Conv2D(1024, (3, 3), activation='relu', strides=2, padding='same', groups=1024, name='conv13a')(x)
    x = Conv2D(1024, (1, 1), activation='relu', padding='valid', name='conv13b')(x)

    # 15th block
    x = AveragePooling2D((1, 1), strides=(1, 1))(x)
    x = Flatten()(x)

    output_tensor = Dense(1000, name='fc')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = MoblieNet((224, 224, 3))
    model.summary()

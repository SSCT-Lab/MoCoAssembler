from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, Dense


def AlexNet(class_num, input_shape):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = Conv2D(64, (11, 11), activation='relu', strides=(4, 4), padding='same', name='conv1')(input_tensor)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # 2nd block
    x = Conv2D(192, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)

    # 3rd block
    x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')(x)

    # 4th block
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)

    # 5th block
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool3')(x)

    # 6th block
    x = AveragePooling2D((1, 1), strides=(1, 1))(x)
    x = Flatten()(x)

    # 7th block
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='linear1')(x)

    # 8th block
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='linear2')(x)

    # output
    output_tensor = Dense(class_num, name='linear3')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == '__main__':
    model = AlexNet(1000, (224, 224, 3))
    model.summary()

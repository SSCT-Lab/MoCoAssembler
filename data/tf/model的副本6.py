from tensorflow.keras import layers, models, Model, Sequential


def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    x = layers.Flatten()(x)                         # output(None, 6*6*128)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dense(num_classes)(x)                  # output(None, 5)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model




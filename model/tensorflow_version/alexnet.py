import tensorflow as tf
from tensorflow import keras


def alexnet(label_num=1000, input_shape=(224, 224, 3)):
    input_tensor = keras.Input(shape=input_shape, dtype="float32")

    x = keras.layers.Conv2D(filters=64, kernel_size=(11, 11), activation='relu', strides=(4, 4), padding='same')(
        input_tensor)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = keras.layers.Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=4096, activation='relu')(x)

    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=8192)])

    with tf.device("/GPU:0"):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train[:100]
        y_train = y_train[:100]

        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = alexnet(10, (32, 32, 3))
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=1)


if __name__ == "__main__":
    go()

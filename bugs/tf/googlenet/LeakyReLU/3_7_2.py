import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.paths import DATASETS_PATH


def googlenet(num_class=1000, input_shape=(224, 224, 3)):
# googlenet input layer
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
# googlenet hidden layer
    x = keras.layers.AveragePooling2D(strides=(3, 1), padding="same")(input_tensor)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=-0.5078522737268056)(x)
# googlenet output layer
    output_tensor = keras.layers.Dense(units=num_class, activation="softmax")(keras.layers.Flatten()(x))

    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    imagenet = np.load(DATASETS_PATH / "imagenet.npz")
    print(imagenet.__dict__)
    x_train = imagenet['x_test'][:100]
    y_train = imagenet['y_test'][:100]

    model = googlenet(1000, (224, 224, 3))
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=1)
    return model.count_params()


if __name__ == "__main__":
    go()

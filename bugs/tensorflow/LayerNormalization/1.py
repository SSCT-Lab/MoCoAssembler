import numpy as np
import tensorflow as tf
from tensorflow import keras
from moco_tf.config.paths import DATASETS_PATH


def resnet18(class_num=1000, input_shape=(224, 224, 3)):
    # input layers
    input_tensor = keras.Input(shape=input_shape, dtype="float32")
    x = input_tensor

    # hidden layers
    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', use_bias=False)(x)
    x = keras.layers.Softmax()(x)
    x = keras.layers.MaxPooling2D(strides=6, padding="same")(x)
    x = InceptionA_102(x=x, in_channels=64, out_channels=64, stride=1)
    x = InceptionA_148(x=x, in_channels=64, out_channels=64, stride=1)
    x = InceptionB_373(x=x, in_channels=64, out_channels=128, stride=2)
    x = InceptionA_806(x=x, in_channels=128, out_channels=128, stride=1)
    x = InceptionB_1584(x=x, in_channels=128, out_channels=256, stride=2)
    x = InceptionA_3438(x=x, in_channels=256, out_channels=256, stride=1)
    x = InceptionB_9110(x=x, in_channels=256, out_channels=512, stride=2)
    x = InceptionA_17836(x=x, in_channels=512, out_channels=512, stride=1)
    x = keras.layers.LayerNormalization(epsilon=0.0, name=None)(x)

    # output layers
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=class_num, activation='softmax')(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

def InceptionA (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)(x)
    branch1 = keras.layers.BatchNormalization(num_features=out_channels)(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization(num_features=out_channels)(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionB (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)(x)
    branch1 = keras.layers.BatchNormalization(num_features=out_channels)(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization(num_features=out_channels)(branch1)
    branch2 = keras.layers.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)(x)
    branch2 = keras.layers.BatchNormalization(num_features=out_channels)(branch2)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    branch2 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch2)
    x = keras.layers.add([branch1, branch2])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionA_102 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization()(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionA_148 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization(center=False)(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionB_373 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)(branch1)
    branch1 = keras.layers.Softmax()(branch1)
    branch2 = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, use_bias=False)(x)
    branch2 = keras.layers.BatchNormalization()(branch2)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    branch2 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch2)
    x = keras.layers.add([branch1, branch2])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionA_806 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU(max_value=0.4867245190840199)(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization()(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionB_1584 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization(scale=False)(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False, activation='sigmoid')(branch1)
    branch1 = keras.layers.BatchNormalization(axis=3)(branch1)
    branch2 = keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=1, strides=stride, use_bias=False)(x)
    branch2 = keras.layers.BatchNormalization()(branch2)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    branch2 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch2)
    x = keras.layers.add([branch1, branch2])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionA_3438 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)(branch1)
    branch1 = keras.layers.BatchNormalization()(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU()(x)

    # output layers
    return x

def InceptionB_9110 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU()(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=True)(branch1)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch2 = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, use_bias=False, activation='softsign')(x)
    branch2 = keras.layers.BatchNormalization(moving_variance_initializer='ones')(branch2)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    branch2 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch2)
    x = keras.layers.add([branch1, branch2])
    x = keras.layers.ReLU(max_value=0.5670042124680282)(x)

    # output layers
    return x

def InceptionA_17836 (x, in_channels, out_channels, stride):

    # hidden layers
    branch1 = keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    branch1 = keras.layers.BatchNormalization()(branch1)
    branch1 = keras.layers.ReLU(threshold=0.060365965387310316)(branch1)
    branch1 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same", use_bias=False, activation='deserialize')(branch1)
    branch1 = keras.layers.BatchNormalization()(branch1)
    # reshape layer
    target_height = x.shape[1]
    target_width = x.shape[2]
    branch1 = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(branch1)
    x = keras.layers.Lambda(lambda _: tf.image.resize(_, (target_height, target_width)))(x)
    x = keras.layers.add([branch1, x])
    x = keras.layers.ReLU(max_value=0.1935564604050818)(x)

    # output layers
    return x


def go():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=8192)])

    with tf.device("/GPU:0"):
        imagenet = np.load(DATASETS_PATH / "imagenet.npz")

    x_train = imagenet['x_test'][:100]
    y_train = imagenet['y_test'][:100]
    x_test = imagenet["x_test"][:50]
    y_test = imagenet["y_test"][:50]

    model = resnet18(1000, (224, 224, 3))
    model.compile(optimizer=keras.optimizers.legacy.SGD(learning_rate=0.3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

    model.evaluate(x_test, y_test, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()
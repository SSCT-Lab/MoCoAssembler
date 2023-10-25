import tensorflow as tf


def alexnet_1():
    network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(48, kernel_size=11, strides=4, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 27*27*48
        tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 13*13*128
        tf.keras.layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
        tf.keras.layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 6*6*128
        tf.keras.layers.Flatten(),  # 6*6*128=4608
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(5)
    ])
    return network

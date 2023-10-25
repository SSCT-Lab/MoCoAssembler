import tensorflow as tf


def FirstNet():
    network = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)  # 输出层
    ])
    return network

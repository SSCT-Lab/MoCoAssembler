import tensorflow as tf
from tensorflow.keras import Sequential


network = Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # 输出层
])
network.build(input_shape=(None, 28*28))

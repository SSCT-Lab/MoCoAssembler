import tensorflow as tf
from tensorflow.keras import Sequential


# 2.网络搭建
network = Sequential([
    # 第一层
    tf.keras.layers.Conv2D(48, kernel_size=11, strides=4, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),
    # 55*55*48
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 27*27*48
    # 第二层
    tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]], activation='relu'),
    # 27*27*128
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 13*13*128
    # 第三层
    tf.keras.layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
    # 13*13*192
    # 第四层
    tf.keras.layers.Conv2D(192, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
    # 13*13*192
    # 第五层
    tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], activation='relu'),
    # 13*13*128
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),  # 6*6*128
    tf.keras.layers.Flatten(),  # 6*6*128=4608
    # 第六层
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    # 第七层
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    # 第八层（输出层）
    tf.keras.layers.Dense(5)
])
network.build(input_shape=(32, 224, 224, 3))
network.summary()

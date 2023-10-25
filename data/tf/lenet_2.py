import tensorflow as tf


def lenet_2():
    network = tf.keras.Sequential([  # 搭建网络容器
        tf.keras.layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3*1卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层，卷积核2*2，步长2
        tf.keras.layers.ReLU(),  # 激活函数
        tf.keras.layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层，16个3*3*6卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层
        tf.keras.layers.ReLU(),  # 激活函数
        tf.keras.layers.Flatten(),  # 拉直，方便全连接层处理

        tf.keras.layers.Dense(120, activation='relu'),  # 全连接层，120个节点
        tf.keras.layers.Dense(84, activation='relu'),  # 全连接层，84个节点
        tf.keras.layers.Dense(10)  # 输出层，10个节点
    ])
    return network
import tensorflow as tf
from tensorflow.keras import Sequential


network = Sequential([
    # 第一层
    tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    # 第二层
    tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    # 第三层
    tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第四层
    tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    # 第五层
    tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第六层
    tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    # 第七层
    tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第八层
    tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),  # 拉直 7*7*512
    # 第九层
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    # 第十层
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    # 第十一层
    tf.keras.layers.Dense(5, activation='softmax')
])
network.build(input_shape=(None, 224, 224, 3))
network.summary()

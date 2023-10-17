import tensorflow as tf

LeNet5_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

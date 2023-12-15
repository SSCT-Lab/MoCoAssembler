import tensorflow as tf

def pointnet(input_shape):
    # input layers
    input_tensor = tf.keras.Input(shape=input_shape, dtype="float32")
    x = input_tensor

    # hidden layers
    x = tf.keras.layers.Conv1D(kernel_size=1, filters=64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(kernel_size=1, filters=128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(kernel_size=1, filters=1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=256)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=10)(x)

    # output layers
    output_tensor = tf.keras.layers.Flatten()(tf.keras.layers.Dense(units=10, activation='softmax')(x))
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    model = pointnet(input_shape=(5, 3))
    x = tf.random.normal(shape=(1,) + (5, 3))
    y = model(x)
    return model



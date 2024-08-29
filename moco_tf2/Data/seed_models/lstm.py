import tensorflow as tf


def lstm(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(input_tensor)
    x = tf.keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(units=16, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Dense(units=1, activation='relu')(x)

    output_tensor = tf.keras.layers.Dense(units=25, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

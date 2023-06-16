import tensorflow as tf
from tensorflow import keras


def lstm(shape=16):
    input_tensor = keras.Input(shape=shape)
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8)(input_tensor)
    x = keras.layers.LSTM(units=16, activation='sigmoid')(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(units=16, activation='relu')(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    model = lstm(16)
    model.summary()


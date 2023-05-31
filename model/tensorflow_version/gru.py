import tensorflow as tf
from tensorflow import keras


def GRU(shape):
    input_tensor = keras.Input(shape=shape)
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8)(input_tensor)
    x = keras.layers.GRU(units=32, dropout=0.5, return_sequences=True)(x)
    x = keras.layers.GRU(units=32, dropout=0.5)(x)
    x = keras.layers.Dense(units=32)(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=1, activation="relu")(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    model = GRU(16)
    model.summary()

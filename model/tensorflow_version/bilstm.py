import tensorflow as tf
from tensorflow import keras


def BiLSTM(shape):
    input_tensor = keras.Input(shape=shape)
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8)(input_tensor)
    x = keras.layers.Dropout(rate=0.1)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    model = BiLSTM(16)
    model.summary()


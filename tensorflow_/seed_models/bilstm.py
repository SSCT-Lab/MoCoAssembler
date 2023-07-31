import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow import keras
from tensorflow_.config.paths import DATASETS_PATH


def bilstm(num_units=25, input_shape=10):
    input_tensor = keras.Input(shape=input_shape)

    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8)(input_tensor)
    x = keras.layers.Dropout(rate=0.1)(x)
    y = keras.layers.LSTM(units=num_units, return_sequences=False, recurrent_dropout=0.1)
    x = keras.layers.Bidirectional(layer=y)(x)
    x = keras.layers.Dense(units=32)(x)
    x = keras.layers.Dropout(rate=0.5)(x)

    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    data = read_csv(DATASETS_PATH / "DIS.csv", header=None, index_col=None, delimiter=",")

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = data[5].values.reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)

    labels = data.iloc[:, -1].values

    train_size = int(len(dataset) * 0.5)
    train_data = dataset[:train_size]
    train_labels = labels[:train_size]

    X, Y = [], []
    for i in range(len(train_data) - 10):
        X.append(train_data[i:i + 10])
        Y.append(train_labels[i + 10])

    X_train, Y_train = np.array(X), np.array(Y)

    model = bilstm(25, 10)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, Y_train, batch_size=8, epochs=1, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow import keras
from config.paths import DATASETS_PATH


def lstm(num_units=25, input_shape=10):
# lstm input layer
    input_tensor = keras.Input(shape=input_shape)
# lstm hidden layer
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8, embeddings_constraint=None)(input_tensor)
    x = keras.layers.LSTM(units=16, activation='sigmoid', unroll=False)(x)
    x = keras.layers.ActivityRegularization(l2=0.6735793371816836, l1=-0.08843623213517704)(x)
# lstm output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=num_units, activation="relu")(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    data = read_csv(DATASETS_PATH / "DIS.csv", header=None, index_col=None, delimiter=',')

    dataset = data[5].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.5)
    train = dataset[:train_size, :]

    X, Y = [], []
    for i in range(len(train) - 10):
        X.append(train[i:i+10, 0])
        Y.append(train[i+10, 0])

    X_train, Y_train = np.array(X), np.array(Y)

    model = lstm(25, 10)
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, Y_train, batch_size=8, epochs=1, verbose=1)

    return model.count_params()


if __name__ == "__main__":
    go()


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow import keras
from tf.config.paths import DATASETS_PATH


def lstm(num_units=25, input_shape=10):
    input_tensor = keras.Input(shape=input_shape)
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8)(input_tensor)
    x = keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(x)
    x = keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(x)
    x = keras.layers.LSTM(units=16, activation='sigmoid', return_sequences=True)(x)
    x = keras.layers.LSTM(units=16, activation='sigmoid')(x)
    x = keras.layers.Dropout(rate=0.1)(x)
    x = keras.layers.Dense(units=1, activation='relu')(x)

    output_tensor = keras.layers.Dense(units=num_units, activation='softmax')(x)
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def go():
    data = read_csv(DATASETS_PATH / "DIS.csv", header=None, index_col=None, delimiter=',')

    dataset = data[5].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.5)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X, Y = [], []
    for i in range(len(train) - 10):
        X.append(train[i:i+10, 0])
        Y.append(train[i+10, 0])

    trainX, trainY = np.array(X), np.array(Y)

    X, Y = [], []
    for i in range(len(test) - 10):
        X.append(test[i:i+10, 0])
        Y.append(test[i+10, 0])

    testX, testY = np.array(X), np.array(Y)

    model = lstm(25, 10)

    model.predict(trainX, verbose=0)
    model.predict(testX, verbose=0)

    return model.count_params()


if __name__ == "__main__":
    go()


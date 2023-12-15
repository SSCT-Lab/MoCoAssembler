import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow import keras
from moco_tf.config.paths import DATASETS_PATH


def pointnet(num_units):
    # input layers
    input_tensor = keras.Input(shape=(None, 10))

    #hidden layers
    x = keras.layers.Conv1D(filters=64, kernel_size=1, padding="valid")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv1D(filters=1024, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(units=512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dense(units=256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # output layers
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=num_units, activation='softmax')(x))
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

    trainX, trainY = np.expand_dims(np.array(X), axis=1), np.expand_dims(np.array(Y), axis=1)

    model = pointnet(25)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(trainX, trainY, epochs=1, verbose=0)

    return model


if __name__ == "__main__":
    go()

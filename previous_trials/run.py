# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 1

from keras.datasets import cifar10
from tensorflow import keras
from keras.optimizers import SGD





# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoing
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def

if __name__ == '__main__':

    from new_model_6 import KitModel
    model=KitModel()
    # 定义优化器
    sgd = SGD(lr=0.2)
    model.compile(
        optimizer=sgd,
        loss='mse',
        metrics=['accuracy'],
    )
    model.summary()
    model.fit(x_train, y_train, batch_size=64, epochs=5)

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)

    print('\ntest loss', loss)
    print('accuracy', accuracy)

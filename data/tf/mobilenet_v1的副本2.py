'''
Simple implementation of mobilenetV1 by tf2.0
'''
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model


def MobilenetV1(input_shape, k, alpha=1, shallow=False):
  '''This function defines a mobilenetv1 architecture.
  '''
  inputs = Input(shape=input_shape)
  x = Conv2D(filters=32 * alpha, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
  x = SeparableConv2D(filters=64 * alpha, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=128 * alpha, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=128 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=256 * alpha, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=256 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)

  if not shallow:
    x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
    x = SeparableConv2D(filters=512 * alpha, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)

  x = SeparableConv2D(filters=1024 * alpha, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)
  x = SeparableConv2D(filters=1024 * alpha, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)

  x = AveragePooling2D(pool_size=(7, 7), strides=1)(x)
  output = Dense(units=k, activation="softmax")(x)

  model = Model(inputs=inputs, outputs=output)

  return model


if __name__ == "__main__":
  model = MobilenetV1(input_shape=(224, 224, 3), k=1000)
  model.summary()
  plot_model(model, to_file='./MobileNetv1.png', show_shapes=True)

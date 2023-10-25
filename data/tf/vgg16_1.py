import tensorflow as tf
from keras import Sequential, Model

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(Model):
    def __init__(self, features, num_classes, input_shape=(32, 32, 3)):
        super(VGG, self).__init__()
        
        self.features = Sequential([
            tf.keras.layers.Input(input_shape),
            features
        ])

        self.classifier = Sequential([
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs, training=False, **kwargs):
        x = self.features(inputs, training=training)
        x = self.classifier(x, training=training)
        return x


def make_layers(cfg):
    nets = []

    for l in cfg:
        if l == 'M':
            nets += [tf.keras.layers.MaxPool2D()]
            continue

        nets += [tf.keras.layers.Conv2D(l, (3, 3), padding='same')]
        nets += [tf.keras.layers.BatchNormalization()]
        nets += [tf.keras.layers.ReLU()]
    return Sequential(nets)


def vgg16_1():
    return VGG(make_layers(cfg['D']), 10)

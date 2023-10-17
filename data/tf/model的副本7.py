from typing import Dict
from copy import deepcopy
import tensorflow as tf
from config import CONFIG_FLOWNET, CONFIG_TRAINING


class MalformedNetworkType(Exception):
    """The provided network type doesn't match one of 'simple' or 'correlation'."""


class FlowNet:
    def __init__(self, config: Dict):
        self.config = config

        self.model = self._construct_network(config)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    @staticmethod
    def get_simple_model(config: Dict) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(384, 512, 6))

        conv_1 = tf.keras.layers.Conv2D(name='conv1', filters=64, kernel_size=7, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(inputs)
        conv_2 = tf.keras.layers.Conv2D(name='conv2', filters=128, kernel_size=5, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(name='conv3', filters=256, kernel_size=5, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(conv_2)
        conv_3_1 = tf.keras.layers.Conv2D(name='conv3_1', filters=256, kernel_size=3, strides=1, padding='same',
                                          activation=tf.keras.activations.relu)(conv_3)
        conv_4 = tf.keras.layers.Conv2D(name='conv4', filters=512, kernel_size=3, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(conv_3_1)
        conv_4_1 = tf.keras.layers.Conv2D(name='conv4_1', filters=512, kernel_size=3, strides=1, padding='same',
                                          activation=tf.keras.activations.relu)(conv_4)
        conv_5 = tf.keras.layers.Conv2D(name='conv5', filters=512, kernel_size=3, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(conv_4_1)
        conv_5_1 = tf.keras.layers.Conv2D(name='conv5_1', filters=512, kernel_size=3, strides=1, padding='same',
                                          activation=tf.keras.activations.relu)(conv_5)
        conv_6 = tf.keras.layers.Conv2D(name='conv6', filters=1024, kernel_size=3, strides=2, padding='same',
                                        activation=tf.keras.activations.relu)(conv_5_1)
        conv_6_1 = tf.keras.layers.Conv2D(name='conv6_1', filters=1024, kernel_size=3, strides=1, padding='same',
                                          activation=tf.keras.activations.relu)(conv_6)

        predict_6 = tf.keras.layers.Conv2D(name='predict_6', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(conv_6_1)

        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation=tf.keras.activations.relu)(conv_6)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation=tf.keras.activations.relu)(predict_6)
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation=tf.keras.activations.relu)(predict_5)
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation=tf.keras.activations.relu)(predict_4)
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation=tf.keras.activations.relu)(predict_3)
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation=tf.keras.activations.relu)(predict_2)
        concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs,
                                  outputs=[predict_6, predict_5, predict_4, predict_3, predict_2, predict_1])

        return tf.keras.Model(inputs=inputs, outputs=predict_1)

    @staticmethod
    def get_corr_model(config: Dict) -> tf.keras.Model:
        raise NotImplementedError("The correlation model hasn't been implemented.")

    @staticmethod
    def _construct_network(config: Dict) -> tf.keras.Model:
        if config['architecture'] == 'simple':
            return FlowNet.get_simple_model(config)
        if config['architecture'] == 'corr':
            return FlowNet.get_corr_model(config)

        raise MalformedNetworkType(f"{config['architecture']}: {MalformedNetworkType.__doc__}")


def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    flownet = FlowNet(config_network)


if __name__ == "__main__":
    main()

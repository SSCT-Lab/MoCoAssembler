"""
"""

from typing import Tuple, Union, Optional, Callable
from collections.abc import Iterable

import tensorflow as tf


class StereoNet(tf.keras.models.Model):
    """ Lifted heavily from https://github.com/zhixuanli/StereoNet
    """

    def __init__(self, k: int = 4, candidate_disparities: int = 192):
        super().__init__(name='')

        self.k = k
        self.disp = (candidate_disparities + 1) // (2**self.k)

        self.feature_extraction = get_downsampling_feature_network(k=self.k)
        self.cost_volume = get_cost_volume_network(k=4)
        self.edge_aware_refinement = EdgeAwareRefinement(k=4)

    def call(self, x, training=False):
        left, right = x
        reference_embedding = self.feature_extraction(left, training=training)
        target_embedding = self.feature_extraction(right, training=training)

        cost = tf.TensorArray(dtype=tf.float32, size=self.disp)
        cost.write(0, reference_embedding - target_embedding)
        for idx in range(1, self.disp):
            cost.write(idx, tf.pad((reference_embedding[:, :, idx:, :] - target_embedding[:, :, :-idx, :]), [[0, 0], [0, 0], [idx, 0], [0, 0]]))
        cost = cost.stack()
        cost = tf.transpose(cost, perm=[1, 0, 2, 3, 4])

        cost = self.cost_volume(cost, training=training)
        cost = tf.squeeze(cost, axis=-1)

        # b, d, h, w = tf.shape(cost)
        disp_initial_l = tf.squeeze(tf.image.resize(tf.reshape(cost, (-1, tf.shape(cost)[2], tf.shape(cost)[3]))[..., tf.newaxis], size=left.shape[1:3]), axis=-1)

        pred_initial_l = tf.reshape(disp_initial_l, (tf.shape(cost)[0], tf.shape(cost)[1], tf.shape(left)[1], tf.shape(left)[2]))

        disp_initial_l = soft_argmin(pred_initial_l, self.disp)

        disp_refined_l = self.edge_aware_refinement(disp_initial_l[..., tf.newaxis], left)

        return disp_refined_l


def soft_argmin(cost_volume, grid_size):
    disp_softmax = tf.keras.layers.Softmax(axis=1)(cost_volume)

    disp_grid = tf.reshape(tf.range(grid_size, dtype=tf.float32), (1, grid_size, 1, 1))
    disp_grid = tf.repeat(tf.repeat(tf.repeat(disp_grid, cost_volume.shape[0], axis=0), cost_volume.shape[2], axis=2), cost_volume.shape[3], axis=3)  # This feels very inelegant

    arg_soft_min = tf.reduce_sum(tf.math.multiply(disp_grid, cost_volume), axis=1)

    return arg_soft_min


class ResBlock(tf.keras.models.Model):
    """ https://www.tensorflow.org/tutorials/customization/custom_layers#models_composing_layers
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]],
                 padding: Union[str, Tuple[int, int], int],
                 dilation_rate: Union[int, Tuple[int, int]],
                 downsampler: Optional[Callable]):
        super().__init__(name='')

        if isinstance(padding, int) or isinstance(padding, Iterable):
            self.conv = tf.keras.models.Sequential()
            self.conv.add(tf.keras.layers.ZeroPadding2D(padding=padding))
            self.conv.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate))
        if isinstance(padding, str):
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU(negative_slope=0.2)

        self.downsampler = downsampler

    def call(self, input_tensor, training=False):
        x = input_tensor

        x = self.conv(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        if self.downsampler is not None:
            input_tensor = self.downsampler(input_tensor)

        return x + input_tensor


def get_downsampling_feature_network(k: int = 4) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, None, 3)))

    for _ in range(k):
        model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid'))

    for _ in range(6):
        model.add(ResBlock(filters=32, kernel_size=(3, 3), strides=(1, 1), padding=1, dilation_rate=(1, 1), downsampler=None))

    model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1)))

    return model


def get_cost_volume_network(k: int = 4) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, None, None, 32)))

    for _ in range(k):
        model.add(tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
        model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU(negative_slope=0.2))

    model.add(tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
    model.add(tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1)))

    return model


class EdgeAwareRefinement(tf.keras.models.Model):
    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.atrous_list = [1, 2, 4, 8, 1, 1]

        self.feature_conv = tf.keras.models.Sequential([tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                                                        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1)),
                                                        tf.keras.layers.ReLU(negative_slope=0.2)
                                                        ])

        self.residual_atrous_net = tf.keras.models.Sequential()
        for dilation_rate in self.atrous_list:
            padding = 1
            if dilation_rate > padding:
                padding = dilation_rate
            self.residual_atrous_net.add(ResBlock(filters=32, kernel_size=3, strides=(1, 1), padding=padding, dilation_rate=dilation_rate, downsampler=None))

        self.final_conv = tf.keras.models.Sequential([tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                                                      tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1)),
                                                      ])

    def call(self, disp, colour):
        # _, _, original_disp_h = tf.shape(disp)
        # disp = tf.image.resize(disp[..., tf.newaxis], size=tf.shape(colour)[1:3])

        # if tf.shape(colour)[2] / original_disp_h >= 1.5:
        # disp *= 8

        output = tf.concat([disp, colour], axis=-1)
        output = self.feature_conv(output)
        output = self.residual_atrous_net(output)

        output = self.final_conv(output)
        output += disp

        output = tf.keras.activations.relu(output)
        # output = tf.squeeze(output, axis=-1)

        return output


def main():
    import numpy as np

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    stereo_net = StereoNet(k=4, candidate_disparities=192)
    left = np.random.random((2, 540, 960, 3))
    right = np.random.random((2, 540, 960, 3))
    out = stereo_net((left, right))

    print('stall')


if __name__ == "__main__":
    main()

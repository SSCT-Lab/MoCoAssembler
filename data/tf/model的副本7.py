""" FlowNet model written in TF2/Keras
    https://arxiv.org/pdf/1504.06852.pdf
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import utils_io as uio
import utils
from config import CONFIG_FLOWNET, CONFIG_TRAINING


class MalformedNetworkType(Exception):
    """The provided network type doesn't match one of 'simple' or 'correlation'."""


class FlowNet:
    """ FlowNetSimple model from the Computer Vision Group of Freiburg.
        https://lmb.informatik.uni-freiburg.de/
        https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf
    """

    def __init__(self, config: Dict):
        self.config = config

        self.model = self._construct_network(config)

    def __getattr__(self, attr):
        """ Rather than potentially override any of the tf.keras.Model methods by subclassing and defining new methods,
            create a composition class with self.model:tf.keras.Model and allow attribute calls directly against self.model
        """
        return getattr(self.model, attr)

    @staticmethod
    def get_simple_model(config: Dict) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(384, 512, 6))

        """ Contracting part """
        conv_1 = tf.keras.layers.Conv2D(name='conv1', filters=64, kernel_size=7, strides=2, padding='same', activation=tf.keras.activations.relu)(inputs)
        conv_2 = tf.keras.layers.Conv2D(name='conv2', filters=128, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(name='conv3', filters=256, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_2)
        conv_3_1 = tf.keras.layers.Conv2D(name='conv3_1', filters=256, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_3)
        conv_4 = tf.keras.layers.Conv2D(name='conv4', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_3_1)
        conv_4_1 = tf.keras.layers.Conv2D(name='conv4_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_4)
        conv_5 = tf.keras.layers.Conv2D(name='conv5', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_4_1)
        conv_5_1 = tf.keras.layers.Conv2D(name='conv5_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_5)
        conv_6 = tf.keras.layers.Conv2D(name='conv6', filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_5_1)
        conv_6_1 = tf.keras.layers.Conv2D(name='conv6_1', filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_6)

        """ The paper itself doesn't have this documented but all implementations, including the original authors, use an extra flow path in the code. """
        predict_6 = tf.keras.layers.Conv2D(name='predict_6', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(conv_6_1)

        """ Expanding part """
        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_6)
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_5)
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_4)
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_3)
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_2)
        concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs, outputs=[predict_6, predict_5, predict_4, predict_3, predict_2, predict_1])

        return tf.keras.Model(inputs=inputs, outputs=predict_1)

    def disable_training(self):
        """ After training is finished, run this method to have self.model predict a single array rather than a list of 6 arrays
        """
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-1].output)

    def enable_training(self):
        """ If you need to re-enable training, run this method to have self.model predict the list of 6 predictions
        """
        output_layers = [layer.output for layer in self.model.layers if 'predict' in layer.name]
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=output_layers)

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


class DataGenerator:
    """ Instantiate then call instance.next_train() to get a generator for training images/labels
            call instance.next_val() to get a generator for validation images/labels
    """

    def __init__(self,
                 network_type: str,
                 flo_normalization: Tuple[float, float],
                 root_path: Path,
                 batch_size: int,
                 validation_batch_size: int,
                 train_ratio: Union[float, int] = 1,
                 test_ratio: Union[float, int] = 0,
                 shuffle: bool = False,
                 augmentations: Optional[Dict] = None):
        self.network_type = network_type

        images = list(root_path.glob('*1.ppm'))
        self.train, self.val, self.test = utils.get_train_val_test(images, train_ratio, test_ratio, shuffle)
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.replace = True
        self.flo_normalization = flo_normalization
        self.augmentations = augmentations

    def next_train(self):

        while True:
            images = np.random.choice(self.train, self.batch_size, replace=self.replace)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            img1 = utils.normalize_images(img1)
            img2 = utils.normalize_images(img2)
            label = utils.normalize_flo(label, self.flo_normalization)

            if not self.augmentations is None:
                img1, img2, label = self._augment(img1, img2, label)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))

    def next_val(self):

        while True:
            images = np.random.choice(self.val, self.validation_batch_size, replace=False)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            img1 = utils.normalize_images(img1)
            img2 = utils.normalize_images(img2)
            label = utils.normalize_flo(label, self.flo_normalization)

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))

    def _augment(self, img1, img2, label):
        # Augmentations are more awkward because of the Siamese architecture, I can't justify applying different color transforms to each image independently
        # I'm 100 certain there is a better way to do this as this is extremely inefficient with each call likely containing some portion of each other call.
        r = np.random.rand(len(self.augmentations))
        r_inc = 0  # This, with r, are used to randomly turn on/off augmentations so that not every augmentation is applied each time
        r_onoff = 2/5
        if 'brightness' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * self.augmentations['brightness']
            def brt(x, idx): return tf.image.adjust_brightness(x, rdm[idx])
            img1 = tf.stack([brt(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([brt(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'multiplicative_colour' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size, 3) * (self.augmentations['multiplicative_colour'][1] -
                                                        self.augmentations['multiplicative_colour'][0]) + self.augmentations['multiplicative_colour'][0]

            def mc(x, idx): return x * rdm[idx]
            img1 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
            img2 = tf.clip_by_value(tf.stack([mc(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
            r_inc += 1
        if 'gamma' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * (self.augmentations['gamma'][1] - self.augmentations['gamma'][0]) + self.augmentations['gamma'][0]
            def gam(x, idx): return tf.image.adjust_gamma(x, gamma=rdm[idx])
            img1 = tf.stack([gam(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([gam(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'contrast' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * (self.augmentations['contrast'][1] - self.augmentations['contrast'][0]) + self.augmentations['contrast'][0]
            def cts(x, idx): return tf.image.adjust_contrast(x, contrast_factor=rdm[idx])
            img1 = tf.stack([cts(im, idx) for idx, im in enumerate(img1)], axis=0)
            img2 = tf.stack([cts(im, idx) for idx, im in enumerate(img2)], axis=0)
            r_inc += 1
        if 'gaussian_noise' in self.augmentations and r[r_inc] <= r_onoff:
            rdm = np.random.rand(self.batch_size) * self.augmentations['gaussian_noise']
            def gau(x, idx): return x + tf.random.normal(x.shape, mean=0.0, stddev=rdm[idx], dtype=x.dtype)
            img1 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img1)], axis=0), clip_value_min=0, clip_value_max=1)
            img2 = tf.clip_by_value(tf.stack([gau(im, idx) for idx, im in enumerate(img2)], axis=0), clip_value_min=0, clip_value_max=1)
            r_inc += 1

        return img1, img2, label


class EndPointError(tf.keras.losses.Loss):
    """ EndPointError is the Euclidean distance between the predicted flow vector and the ground truth averaged over all pixels.
        The resizing is required because the loss is calculated for each flow prediction which occur at different stride levels,
        resizing effectively averages at that scale.
    """

    def call(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(tf.image.resize(y_true, y_pred.shape[1:3]) - y_pred), axis=1, keepdims=True))


def load_images(image_name: str):
    """ Debug function to load the first image for visualization
    """
    root_path = Path(r'C:\Users\andre\Documents\Python\FlowNet_TF2\data\FlyingChairs_release\data')
    flo_path = root_path / f'{image_name}_flow.flo'
    img1_path = root_path / f'{image_name}_img1.ppm'
    img2_path = root_path / f'{image_name}_img2.ppm'
    flo = uio.read(str(flo_path))
    img1 = uio.read(str(img1_path))
    img2 = uio.read(str(img2_path))
    img = np.expand_dims(np.concatenate([img1, img2], axis=-1), axis=0)

    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # ax[0,0].imshow(img1)
    # ax[0,1].imshow(img2)
    # ax[1,0].imshow(flo[...,0])
    # ax[1,1].imshow(flo[...,1])
    # plt.show()

    return img, np.expand_dims(flo, axis=0)


def show_images(simple_images, label):
    """
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].imshow(simple_images[..., :3])
    ax[0, 1].imshow(simple_images[..., 3:])
    ax[1, 0].imshow(label[..., 0])
    ax[1, 1].imshow(label[..., 1])
    plt.show()


def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)

    # On first run, populate the min, max scaling values for the flo dataset
    # min, max = utils.get_training_min_max(config_training['img_path'])

    flownet = FlowNet(config_network)

    loss = EndPointError()

    flownet.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=[loss, loss, loss, loss, loss, loss],
                    loss_weights=config_training['loss_weights'][::-1])

    data_generator = DataGenerator(config_network['architecture'],
                                   config_network['flo_normalization'],
                                   config_training['img_path'],
                                   config_training['batch_size'],
                                   config_training['validation_batch_size'],
                                   config_training['train_ratio'],
                                   config_training['test_ratio'],
                                   config_training['shuffle'],
                                   config_training['augmentations'])

    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = f"checkpoint/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=False,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    if not config_training['pretrained_path'] is None:
        flownet.model = tf.keras.models.load_model(config_training['pretrained_path'], custom_objects={'EndPointError': EndPointError})

    # flownet.fit(x=data_generator.next_train(),
    #             epochs=10,
    #             verbose=1,
    #             steps_per_epoch=22872 // config_training['batch_size'],
    #             validation_data=data_generator.next_val(),
    #             validation_steps=4,
    #             validation_batch_size=config_training['validation_batch_size'],
    #             callbacks=[tensorboard_callback, model_checkpoint_callback],
    #             # use_multiprocessing=True
    #             )
    flownet.disable_training()

    #
    # Temporary debugging and visualization
    #
    img, flo = load_images(image_name="22868")
    norm_img = utils.normalize_images(img)
    predicted_flo = flownet.predict(norm_img)
    predicted_flo = utils.denormalize_flo(predicted_flo, config_network['flo_normalization'])
    predicted_flo = tf.image.resize(predicted_flo, (384, 512)).numpy()

    import matplotlib.pyplot as plt
    scale_min = np.min([np.min(flo), np.min(predicted_flo)])
    scale_max = np.max([np.max(flo), np.max(predicted_flo)])
    fig, ax = plt.subplots(ncols=2, nrows=3)
    ax[0, 0].imshow(img[0, ..., :3])
    ax[0, 1].imshow(img[0, ..., 3:])
    ax[0, 0].set_ylabel('Input images')
    ax[1, 0].imshow(flo[0, ..., 0], vmin=scale_min, vmax=scale_max)
    ax[1, 1].imshow(flo[0, ..., 1], vmin=scale_min, vmax=scale_max)
    ax[1, 0].set_ylabel('Ground truth flows')
    ax[2, 0].imshow(predicted_flo[0, ..., 0], vmin=scale_min, vmax=scale_max)
    ax[2, 1].imshow(predicted_flo[0, ..., 1], vmin=scale_min, vmax=scale_max)
    ax[2, 0].set_ylabel('Predicted flows')
    plt.show()

    print('stall')


if __name__ == "__main__":
    main()

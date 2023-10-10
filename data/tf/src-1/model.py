import tensorflow as tf
from src.layers import *

import gin
from copy import deepcopy


@gin.configurable
class EnsNet(tf.keras.models.Model):
    def __init__(self, config: dict):
        super(EnsNet, self).__init__()
        self.config = config

        self.forward = tf.keras.Sequential()
        for block_config in self.config['block_config']:
            self.forward.add(EnsBlock(**block_config))
        self.main_classifier = Classifier(**self.config['classifier_config'])
        # LATER: Maybe can replace list of classifiers with a single local convolutional classifier?
        self.sub_classifier = [
            Classifier(**self.config['classifier_config']) for _ in range(10)
        ]

    def compile(self, *args, **kwargs):
        super(EnsNet, self).compile(*args, **kwargs)
        self.sub_optimizer = [
            deepcopy(kwargs.get('optimizer')) for _ in range(10)
        ]

    @tf.function
    def get_mode(self, X):
        y, _, count = tf.unique_with_counts(X)
        return y[tf.argmax(count)]

    @tf.function
    def train_step(self, data):
        X, y = data

        # 1. Update CNN
        with tf.GradientTape() as tape:
            feats = self.forward(X, training=True)
            y_hat_main = self.main_classifier(feats, training=True)
            loss_main = self.compiled_loss(y, y_hat_main, regularization_losses=self.losses)
        trainable_vars = self.forward.trainable_variables + self.main_classifier.trainable_variables
        gradients = tape.gradient(loss_main, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 2. Update subnetworks
        y_hat_subs = []
        for feat, classifier, optimizer in zip(tf.split(feats, 10, axis=-1), self.sub_classifier, self.sub_optimizer):
            with tf.GradientTape() as tape:
                y_hat_sub = classifier(feat, training=True)
                loss_sub = self.compiled_loss(y, y_hat_sub, regularization_losses=self.losses)
            gradients = tape.gradient(loss_sub, classifier.trainable_variables)
            optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
            y_hat_subs.append(y_hat_sub)

        y_hat = tf.concat([y_hat_main] + y_hat_subs, axis=0)
        y = tf.concat([y] * 11, axis=0)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        X, y = data
        feats = self.forward(X, training=False)
        y_hat_main = self.main_classifier(feats, training=False)

        y_hat_subs = []
        for feat, classifier in zip(tf.split(feats, 10, axis=-1), self.sub_classifier):
            y_hat_sub = classifier(feat, training=False)
            y_hat_subs.append(y_hat_sub)

        y_hat = tf.stack([y_hat_main] + y_hat_subs, axis=1)
        y_hat_mean = tf.reduce_mean(y_hat, axis=1)
        self.compiled_loss(y, y_hat_mean, regularization_losses=self.losses)
        y_hat_mode = tf.map_fn(self.get_mode, tf.argmax(y_hat, axis=-1))
        y_hat_mode = tf.one_hot(y_hat_mode, depth=10)
        self.compiled_metrics.update_state(y, y_hat_mode)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def call(self, X, training: bool = False):
        feats = self.forward(X, training=training)
        y_hat_main = tf.argmax(self.main_classifier(feats, training=training), axis=-1)
        y_hat_subs = []
        for feat, classifier in zip(tf.split(feats, 10, axis=-1), self.sub_classifier):
            y_hat_sub = tf.argmax(classifier(feat, training=training), axis=-1)
            y_hat_subs.append(y_hat_sub)
        y_hat_total = tf.stack([y_hat_main] + y_hat_subs, axis=-1)
        y_hat_total = tf.map_fn(fn=self.get_mode, elems=y_hat_total)
        return y_hat_total

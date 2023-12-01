import os
import tensorflow as tf
import numpy as np


DATASETS_PATH = "D:/SaveProjects/MoCo(Save)/codes/datasets"


def get_mnist():
    with tf.device("/GPU:0"):
        mnist = np.load(os.path.join(DATASETS_PATH, "MNIST.npz"))
    x_train = mnist['x_train'][:100]
    y_train = mnist['y_train'][:100]
    x_test = mnist["x_test"][:50]
    y_test = mnist["y_test"][:50]

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    return x_train, y_train, x_test, y_test


def get_cifar10():
    with tf.device("/GPU:0"):
        cifar10 = np.load(os.path.join(DATASETS_PATH, "cifar10.npz"))
    x_train = cifar10['x_train'][:100]
    y_train = cifar10['y_train'][:100]
    x_test = cifar10["x_test"][:50]
    y_test = cifar10["y_test"][:50]

    x_train = x_test / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def get_imagenet():
    with tf.device("/GPU:0"):
        imagenet = np.load(os.path.join(DATASETS_PATH, "imagenet.npz"))
    x_train = imagenet['x_test'][:100]
    y_train = imagenet['y_test'][:100]
    x_test = imagenet["x_test"][:50]
    y_test = imagenet["y_test"][:50]

    return x_train, y_train, x_test, y_test

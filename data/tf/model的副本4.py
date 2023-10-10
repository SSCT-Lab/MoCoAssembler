import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, activations
from module import Transform_Net, conv2d, reshape, dense, pooling, lr_decay, set_bn, bn_decay


class Classification(layers.Layer):
    def __init__(self, name, lr):
        super(Classification, self).__init__()
        
        # define hyperparameter
        lr_schedule = lr_decay(lr,
                               decay_steps=6250,
                               decay_rate=0.7,
                               staircase=True,
                               clip=0.00001)
        self.bn_decay = bn_decay(momentum=0.5, 
                                 decay_steps=6250, 
                                 decay_rate=0.5, 
                                 staircase=True, 
                                 clip=0.99)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # define neural network
        # Tnet: [K]
        # conv2d: [filters, kernel, stride, padding, activation, batch norm]
        # dense: [neurons, activation, batch norm, drop out]
        self.net_para = [['{}_Tnet_1'.format(name), [3]],
                    ['{}_matrix_1'.format(name), []],
                    ['{}_reshape1'.format(name), [-1, 3, 1]],
                    ['{}_conv2d_1'.format(name), [64, (1, 3), (1, 1), 'valid', 'relu', True]],
                    ['{}_conv2d_2'.format(name), [64, (1, 1), (1, 1), 'valid', 'relu', True]],
                    ['{}_Tnet_2'.format(name), [64]],
                    ['{}_reshape2'.format(name), [-1, 64]],
                    ['{}_matrix_2'.format(name), []],
                    ['{}_reshape3'.format(name), [-1, 1, 64]],
                    ['{}_conv2d_3'.format(name), [64, (1, 1), (1, 1), 'valid', 'relu', True]],
                    ['{}_conv2d_4'.format(name), [128, (1, 1), (1, 1), 'valid', 'relu', True]],
                    ['{}_conv2d_5'.format(name), [1024, (1, 1), (1, 1), 'valid', 'relu', True]],
                    ['{}_pooling1'.format(name), ['GlobalMaxPooling2D']],
                    ['{}_reshape3'.format(name), [-1]], 
                    ['{}_dense_1'.format(name), [512, 'relu', True, 0.3]],
                    ['{}_dense_2'.format(name), [256, 'relu', True, 0.3]],
                    ['{}_dense_3'.format(name), [40, None, None, None]],
                    ['{}_softmax'.format(name), ['softmax']]]
        self.nets = []
        for layer in self.net_para:
            if "Tnet" in layer[0]:
                self.nets.append(Transform_Net(name=layer[0], K=layer[1][0]))
            elif 'conv2d' in layer[0]:
                self.nets.append(conv2d(layer))
            elif 'dense' in layer[0]:
                self.nets.append(dense(layer))
            elif 'reshape' in layer[0]:
                self.nets.append(reshape(layer))
            elif 'pooling' in layer[0]:
                self.nets.append(pooling(layer))
            elif 'matrix' in layer[0]:
                self.nets.append(layers.Dot(axes=(-1, -1)))
            elif 'softmax' in layer[0]:
                self.nets.append(layers.Activation(activations.softmax))

    def call(self, x):
        trans_matrix = None
        momentum = self.bn_decay(self.opt.iterations)
        for i, layer in enumerate(self.net_para):
            if "Tnet" in layer[0]:
                trans_matrix = self.nets[i](x)
                set_bn(self.nets[i], momentum)
            elif "matrix" in layer[0]:
                x = self.nets[i]([x, trans_matrix])
            elif "dense" in layer[0]:
                x = self.nets[i](x)
                set_bn(self.nets[i], momentum)
            elif "conv2d" in layer[0]:
                x = self.nets[i](x)
                set_bn(self.nets[i], momentum)               
            else:
                x = self.nets[i](x)
        return x, trans_matrix

    def loss(self, pred, label, matrix, alpha): 
        K = matrix.shape[1]
        mat = tf.matmul(matrix, tf.transpose(matrix, perm=[0,2,1]))
        # orthogonal matrix cosntraint
        l2_norm = tf.nn.l2_loss(mat-tf.constant(np.eye(K), dtype=tf.float32))
        return tf.reduce_mean(losses.SparseCategoricalCrossentropy(from_logits=False)(label, pred))+alpha*l2_norm

    def accuracy(self, pred, label):
        pred = tf.argmax(pred, 1)
        correct_predictions = tf.equal(pred, label.reshape(-1))
        acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return acc

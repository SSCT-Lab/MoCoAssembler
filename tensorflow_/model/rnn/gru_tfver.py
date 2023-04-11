# -*- coding: utf-8 -*-

"""
@Title   : TensorFlow implementation of GRU
@Time    : Mar. 13th, 2023
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""

import tensorflow as tf


class GRU_cell(object):

    def __init__(self, input_nodes, hidden_unit, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes

        self.Wx = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))

        self.Wr = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.br = tf.Variable(tf.random.truncated_normal([self.hidden_unit], mean=1))

        self.Wz = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.bz = tf.Variable(tf.random.truncated_normal([self.hidden_unit], mean=1))

        self.Wh = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))

        self.Wo = tf.Variable(tf.random.truncated_normal([self.hidden_unit, self.output_nodes], mean=1, stddev=.01))
        self.bo = tf.Variable(tf.random.truncated_normal([self.output_nodes], mean=1, stddev=.01))

        self._inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, None, self.input_nodes], name='inputs')

        batch_input_ = tf.transpose(self._inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(batch_input_)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

    def Gru(self, previous_hidden_state, x):
        z = tf.sigmoid(tf.matmul(x, self.Wz) + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + self.br)

        h_ = tf.tanh(tf.matmul(x, self.Wx) +
                     tf.matmul(previous_hidden_state, self.Wh) * r)

        current_hidden_state = tf.multiply((1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state


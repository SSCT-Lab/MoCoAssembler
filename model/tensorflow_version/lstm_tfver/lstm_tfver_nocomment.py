# https://www.twblogs.net/a/5b7ce4952b71770a43dd0abb

import numpy as np
import tensorflow as tf


class LSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.output_size = output_size

        with tf.name_scope("inputs"):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, self.input_size], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, n_steps, self.output_size], name="ys")
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.name_scope("cost"):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(0.006).minimize(self.cost)

    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name="weights"):
        initializer = tf.random_normal_initializer(mean=0, stddev=1, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name="biases"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        bs_in = self._bias_variable([self.cell_size, ])
        with tf.name_scope("Wx_puls_b"):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("initial_state"):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32, )
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y,
                                                                     initial_state=self.cell_init_state,
                                                                     time_major=False)

    def add_output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="2_2D")
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        with tf.name_scope("Wx_plus_b"):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name="reshape_pred")],
            [tf.reshape(self.ys, [-1], name="reshape_target")],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name="losses"
        )
        with tf.name_scope("average_cost"):
            self.cost = tf.div(tf.reduce_sum(losses, name="losses_sum"), self.batch_size, name="average_cost")
            tf.summary.scalar("cost", self.cost)


if __name__ == "__main__":
    model = LSTM(20, 1, 1, 10, 50)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    sess.run(init)

    BATCH_START = 0
    for i in range(200):
        xs = np.arange(BATCH_START, BATCH_START + 1000).reshape((50, 20)) / (10 * np.pi)
        seq = np.sin(xs)[:, :, np.newaxis]
        res = np.cos(xs)[:, :, np.newaxis]
        BATCH_START += 20

        if i == 0:
            feed_dict = {model.xs: seq, model.ys: res}
        else:
            feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}

        _, cost, state, pred = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred],
                                        feed_dict=feed_dict)

        if i % 20 == 0:
            print("cost:", round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

"""
https://www.twblogs.net/a/5b7ce4952b71770a43dd0abb
用自己創建的 sin 曲線預測一條 cos 曲線
PS：深度學習中經常看到epoch、 iteration和batchsize，下面按自己的理解說說這三個的區別：
（1）batchsize：批大小。在深度學習中，一般採用SGD訓練，即每次訓練在訓練集中取batchsize個樣本訓練；
（2）iteration：1個iteration等於使用batchsize個樣本訓練一次；
（3）epoch：1個epoch等於使用訓練集中的全部樣本訓練一次；
舉個例子，訓練集有1000個樣本，batchsize=10，那麼：
訓練完整個樣本集需要：
100次iteration，1次epoch。
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 定義超參數
BATCH_START = 0  # 建立 batch data 時候的 index
TIME_STEPS = 20  # time_steps也就是n_steps,等於序列的長度
BATCH_SIZE = 50  # 批次的大小
INPUT_SIZE = 1  # sin 數據輸入 size
OUTPUT_SIZE = 1  # cos 數據輸出 size
CELL_SIZE = 10  # 隱藏層規模
LR = 0.006  # 學習率


# 1. 使用tf.Variable()的時候，tf.name_scope()和tf.variable_scope() 都會給 Variable 和 op 的 name屬性加上前綴。
# 2. 使用tf.get_variable()的時候，tf.name_scope()就不會給 tf.get_variable()創建出來的Variable加前綴。

# 定義 LSTM 的主體結構
class LSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.output_size = output_size
        # 初始化幾個函數
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
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # 定義三個變量
    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name="weights"):
        initializer = tf.random_normal_initializer(mean=0, stddev=1, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name="biases"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    # 接下來定義幾個函數
    def add_input_layer(self):
        # 應該我們只能在二維數據上矩陣相乘，計算logits,之後在reshape成3維。以下同理
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # 輸入shape(batch_size*n_step,input_size)
        # 權重的shape(in_size,cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # 偏置的shape (cell_size,)
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps ,cell_size)
        with tf.name_scope("Wx_puls_b"):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y-->>=(batch,n_steps,cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("initial_state"):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32, )
        # 如果l_in_y的shape是(n_steps,batch,cell_size)的話，則對應的time_major=True
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape=(batch * n_steps,cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="2_2D")
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        with tf.name_scope("Wx_plus_b"):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # shape=(batch*n_steps,output_size)

    def compute_cost(self):
        # 計算一個batch內每一樣本的loss
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name="reshape_pred")],  # 平鋪一下維數
            [tf.reshape(self.ys, [-1], name="reshape_target")],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name="losses"
        )
        with tf.name_scope("average_cost"):
            # 計算每一個batch的平均loss，因爲梯度更新是在計算一個batch的平均誤差的基礎上進行更新的
            self.cost = tf.div(tf.reduce_sum(losses, name="losses_sum"), self.batch_size, name="average_cost")
            tf.summary.scalar("cost", self.cost)


# 數據生成
# 生成一個批次大小的數據的 get_batch function:
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs的shape是（50batch,20steps）
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (
            10 * np.pi)  # 定義x
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # 返回seq，res 的shape(batch,step,1),xs的shape爲(batch_size,time_steps)
    # 一般像這種[:,:,np.newaxis]叫做擴維技術，從2爲變成3維,擴的維數爲1。
    # seq[:, :, np.newaxis].shape=（50,20,1）
    # plt.plot (xs, seq, 'r-', xs, res, 'b-')
    # plt.show ()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


if __name__ == "__main__":
    model = LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # 版本控制
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on
    # 在terminal中輸入$ tensorboard --logdir='logs'，讓後在瀏覽器中Chrome (http://0.0.0.0:6006/)查看tensorboard
    plt.ion()
    plt.show()
    for i in range(200):  # 訓練200次，訓練一次一個batch
        seq, res, xs = get_batch()  # 此時的seq,res都是3維數組，shape=(batch,time_steps,1),這裏的1就是input_size
        if i == 0:
            feed_dict = {model.xs: seq,
                         model.ys: res
                         # 創建初始狀態，這裏就開始體現類的優勢了，直接調用裏面的xs,ys,
                         }
        else:
            feed_dict = {model.xs: seq,
                         model.ys: res,
                         model.cell_init_state: state  # 用最後的state代替初始化的state
                         }
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)
        # 輸出值和帶入的參數順序一一對應，cost對應model.cost,等等

        # xs[0,:],表示的是一個batch裏面的第一個序列，因爲xs是由np.arange()函數生成的，
        # 所以xs在對於每一個batch來說，同一個batch裏面的每個序列都是一樣的
        # 例如xs的batch_size=3,time_step=4,[[0,1,2,3],
        #                                 [0,1,2,3],
        #                                 [0,1,2,4]],shape=(3,4)
        # res[0].flatten()表示的是一個batch裏面的第一個序列，序列長度爲time_steps * 1
        plt.plot(xs[0, :], res[0].flatten(), "r", xs[0, :], pred.flatten()[:TIME_STEPS], "b--")
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:  # 每訓練20個批次來打印一次當時的cost
            print("cost:", round(cost, 4))  # 輸出每一個batch的平均cost，約到零後面4位小數點
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

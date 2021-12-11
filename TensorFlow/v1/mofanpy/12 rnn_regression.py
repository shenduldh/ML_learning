import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

tf.disable_v2_behavior()


# 任务: 翻译序列
# ① 数据集: 给定序列 (输入) 和该序列被翻译后的序列 (标签)
# ② 模型目标: 要求输入给定序列后, 可以返回与标签序列相同的序列
# 也就是输入一个序列 (序列中每个数据按时间点顺序输入),
# 输出一个序列 (由各个时间点的输出组成)。
# 在可视化中, 都是一段一段的标签序列和由模型输出的翻译序列的对比,
# 它们之间并不是连续的, 只是通过处理让他们看起来像是连续的。


BATCH_START = 0
BATCH_SIZE = 50
CELL_SIZE = 10
LEARNING_RATE = 0.006
# 一个序列中有多少个数据 (序列大小/长度), 或称TIME_STEPS
SEQ_SIZE = 20
# (一个时间点上) 输入数据有多长 (维度、特征数)
INPUT_SIZE = 1
# (一个时间点上) 输出数据有多长 (维度、特征数)
OUTPUT_SIZE = 1


def get_batch():
    global BATCH_START, SEQ_SIZE
    xs = np.arange(BATCH_START, BATCH_START+SEQ_SIZE *
                   BATCH_SIZE).reshape((BATCH_SIZE, SEQ_SIZE))/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += SEQ_SIZE

    # shape(batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None):
    # logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    # targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    # weights: List of 1D batch-sized float-Tensors of the same length as logits.
    # return: log_pers, 形状是[batch_size].

    log_perp_list = []

    for logit, target, weight in zip(logits, targets, weights):
        if softmax_loss_function is None:
            target = array_ops.reshape(target, [-1])
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                logit, target)
        else:
            crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

    log_perps = math_ops.add_n(log_perp_list)

    if average_across_timesteps:
        total_size = math_ops.add_n(weights)
        total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
        log_perps /= total_size

    return log_perps


class LSTMRNN(object):
    def __init__(self, seq_size, input_size, output_size, cell_size, batch_size):
        self.seq_size = seq_size
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        self.xs = tf.placeholder(
            tf.float32, [batch_size, seq_size, input_size])
        self.ys = tf.placeholder(
            tf.float32, [batch_size, seq_size, output_size])

        self.add_input_layer()
        self.add_cell()
        self.add_output_layer()

        self.comput_cost()
        self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def add_input_layer(self):
        inputs = tf.reshape(self.xs, [-1, self.input_size])
        weights = self.weight_variable([self.input_size, self.cell_size])
        biases = self.bias_variable([self.cell_size, ])
        WX_plus_b = tf.matmul(inputs, weights)+biases
        self.input_layer_y = tf.reshape(
            WX_plus_b, [-1, self.seq_size, self.cell_size])

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell.zero_state创建了[c_state, h_state]
        # c_state表示初始时间点的memory值, h_state表示初始时间点的"input"值
        # c_state: shape(batch_size, cell_size)
        # 为batch中的每个样本分别创建长度为cell_size的初始状态
        # 一个样本对应cell_size个LSTM单元, 每个单元都需要初始一个状态
        # 即样本间不共享状态, LSTM单元间不共享状态
        self.cell_init_state = lstm_cell.zero_state(
            self.batch_size, dtype=tf.float32)
        # self.cell_outputs.shape=(50, 20, 10)
        # 即50表示有50个样本, 每个样本包含所有时间点(共20个)的输出,
        # 每个时间点的输出长度就是cell的个数(共10个)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.input_layer_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        inputs = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        weights = self.weight_variable([self.cell_size, self.output_size])
        biases = self.bias_variable([self.output_size, ])
        # self.prediction.shape=(batch_size*seq_size, output_size)
        self.prediction = tf.matmul(inputs, weights)+biases

    def comput_cost(self):
        losses = sequence_loss_by_example(
            [tf.reshape(self.prediction, [-1])],
            [tf.reshape(self.ys, [-1])],
            [tf.ones([self.batch_size*self.seq_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error
        )
        self.cost = tf.div(tf.reduce_sum(losses), self.batch_size)

    def ms_error(self, target, prediction):
        return tf.square(prediction-target)

    def weight_variable(self, shape):
        initial_value = tf.random_normal(mean=0., stddev=1., shape=shape)
        return tf.Variable(initial_value)

    def bias_variable(self, shape):
        initial_value = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value)


if __name__ == '__main__':
    model = LSTMRNN(SEQ_SIZE, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.show()

    for i in range(200):
        seq, res, xs = get_batch()

        if(i == 0):
            feed_dict = {
                model.xs: seq,
                model.ys: res
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                # 该例子将上一批样本的状态延续到下一批样本
                model.cell_init_state: state
            }

        # tensorflow应该是在run的时候去寻找对应的变量来进行赋值
        # 因此feed_dict的键采用对应变量的引用
        _, cost, state, pred = sess.run(
            [model.train, model.cost, model.cell_final_state, model.prediction],
            feed_dict=feed_dict
        )

        # TODO: prediction的形状?
        # 可视化翻译过程 (每次只看每批数据的第一个样本序列的预测情况)
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :],
                 pred.flatten()[:SEQ_SIZE], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost:', round(cost, 4))

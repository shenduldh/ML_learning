import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#########################
## 目标: 预测图片中的数字 ##
########################

# NOTE:
# RNN 是基于序列的模型, 因此把图片的每一行当作是序列
# 模型结构:
#   input hidden layer
#   memory cell (LSTM)
#   output hidden layer
# 使得该网络成为 RNN 的, 就是 memory cell 这一层
# 也就是说, RNN 之所以是 RNN, 是因为它网络中存在具有记忆的网络层


# 在此处定义模型的所有层
def RNN(X, weights, biases):
    """
        X: shape(128,28,28)
        weights: {
            'in': shape(28,128),
            'out': shape(128,10)
        }
        biases: {
            'in': shape(128,),
            'out': shape(10,)
        }
    """

    # input hidden layer
    X = tf.reshape(X, [-1, n_inputs])  # X: shape(128*28,28)
    # X_input: shape(128*28,128)
    X_input = tf.matmul(X, weights['in'])+biases['in']
    # X_input: shape(128,28,128)
    X_input = tf.reshape(X_input, [-1, n_steps, n_lstm_cells])

    # memory cell (LSTM)
    # 此处的 cell 接收整个序列作为输入,
    # 然后按序处理序列中的每一个时间点,
    # 最后将最后一个时间点的输出作为整个 cell 的输出,
    # 也就是 output hidden layer 的输入
    # tf.nn.rnn_cell.BasicLSTMCell:
    #       创建一层有n_lstm_cells个cell的LSTM层
    #       forget_bias指forget gate的bias
    #       state_is_tuple=True: 用元组表示每个时间点输出的状态 (c_state,h_state)
    #       c_state是当前时间点的memory值, h_sate是当前时间点的output值
    #       c_state、h_state和实际输入共同构成下一个时间点的输入
    # lstm_cells.zero_state: 用0初始化状态, 并将初始化的状态返回
    lstm_cells = tf.nn.rnn_cell.BasicLSTMCell(
        n_lstm_cells, forget_bias=1.0, state_is_tuple=True)
    initial_state = lstm_cells.zero_state(batch_size, dtype=tf.float32)
    # 获取序列最后一个时间点的输出的方法:
    #  方法一:
    #       因为我们想要的只是最后一个时间点的输出,
    #       因此手动执行整个过程, 然后将最后一个输出outputs[-1]传递给下一层
    # outputs = []
    # state = initial_state
    # # 定义变量空间, 避免破坏外部的同名变量
    # with tf.variable_scope('RNN') as scope:
    #     for time_step in range(n_steps):
    #         if time_step > 0:
    #             # 重用变量是为了权值共享,
    #             # 即每个时间点所使用的权值参数都是同一个,
    #             # 通过重用就可以让每个时间点都使用同一个权值参数变量
    #             scope.reuse_variables()
    #         # X_input[:, time_step, :]: 用于取出每个时间点上的输入数据
    #         (output, state) = lstm_cells(X_input[:, time_step, :], state)
    #         outputs.append(output)
    #
    # # output hidden layer
    # results = tf.matmul(outputs[-1], weigths['out'])+biases['out']

    # 方法二:
    #       tf.nn.dynamic_rnn其实就是上面的自动版本
    #       time_major=False: 表示输入序列的时间点在X_input的第1个轴上
    #       time_major=True: 表示输入序列的时间点在X_input的第0个轴上
    #       因为是将整个序列直接输入进来, 因此要有一个轴来区分序列中不同时间点的数据
    outputs, states = tf.nn.dynamic_rnn(
        lstm_cells, X_input, initial_state=initial_state, time_major=False)

    # 这里输出的states是最后一个时间点的状态, 因此states[1]就代表了最后一个时间点的输出
    # final_output =states[1]
    # 此外, 由于outputs保存了每个时间点的输出, 因此也可以从outputs中取出最后一个时间点的输出
    # tf.transpose将outputs矩阵进行转置
    # tf.unstack将转置后的outputs进行分解, 并将分解后的结果组成list返回
    # 此时, outputs[-1]就是最后一个时间点的输出
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # LSTM有128个cell, 则一个样本有128个输出
    final_output = outputs[-1]  # shape(128,128)

    # output hidden layer
    results = tf.matmul(
        final_output, weigths['out'])+biases['out']  # shape(128,10)

    return results


# hyperparameters
learning_rate = 0.001  # LSTM的学习率要尽量低
training_iterations = 100000
batch_size = 128

# other information
n_inputs = 28
n_steps = 28
n_lstm_cells = 128
n_classes = 10


#############
## 构建模型 ##
############

# 输入和输出
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 两层 hidden layer 的参数
weigths = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_lstm_cells])),
    'out': tf.Variable(tf.random_normal([n_lstm_cells, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_lstm_cells, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

prediction = RNN(x, weigths, biases)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

is_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


#############
## 训练模型 ##
############

# 使用 Google 提供的数据集 MNIST
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step*batch_size < training_iterations:
        # MiniBatch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        sess.run(train, feed_dict={
            x: batch_x,
            y: batch_y
        })
        if step % 50 == 0:
            accuracy_value = sess.run(accuracy, feed_dict={
                x: batch_x,
                y: batch_y
            })
            print(accuracy_value)
        step += 1

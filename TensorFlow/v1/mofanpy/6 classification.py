import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#########################
## 目标: 预测图片中的数字 ##
########################


def add_layer(inputs, in_size, out_size, activator=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activator is None:
        outputs = Wx_plus_b
    else:
        outputs = activator(Wx_plus_b)
    return outputs


def compute_accuracy(x, y):
    global prediction
    prediction_value = sess.run(prediction, feed_dict={xs: x})
    # tf.argmax(input,axis): 根据axis取值的不同返回每行或者每列中最大元素的索引
    is_correct = tf.equal(tf.argmax(prediction_value, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # tensorflow1.0的运算函数都不是即时运行的，需要用Session来启动
    result = sess.run(accuracy)
    return result


#############
## 构建模型 ##
############

xs = tf.placeholder(tf.float32, [None, 784])  # 28×28
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activator=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))

train = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

initializer = tf.global_variables_initializer()


#############
## 训练模型 ##
############

sess = tf.Session()
sess.run(initializer)


# 使用 Google 提供的数据集
# input_data.read_data_sets函数就是加载数据集 (如果没有的话就会从网上下载数据集)
# 数据集介绍:
#   输入: 28×28的手写数字图片
#   输出: 数字1-10的概率分布
#   标签: 数字1-10的 one_hot_vector
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

for step in range(2001):
    # 一次训练只取出部分数据, 即 MiniBatch
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={
        xs: batch_xs,
        ys: batch_ys
    })
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

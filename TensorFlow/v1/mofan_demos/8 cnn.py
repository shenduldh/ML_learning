import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#########################
## 目标: 预测图片中的数字 ##
########################


def compute_accuracy(x, y):
    global prediction
    prediction_value = sess.run(prediction, feed_dict={xs: x, keep_prob: 1})
    is_correct = tf.equal(tf.argmax(prediction_value, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    result = sess.run(accuracy)
    return result


def add_layer(inputs, in_size, out_size, activator=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activator is None:
        outputs = Wx_plus_b
    else:
        outputs = activator(Wx_plus_b)
    return outputs


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # x是图片的所有参数, W是此卷积层的权重
    # strides为步长, [0]和[3]是默认值, [1]和[2]分别代表窗口在x和y方向运动1步
    # 一次一步则图像的长宽保持原样, 相当于在每个像素点上都采样一次
    # padding采用的方式是SAME
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化窗口（或采样范围）的尺寸大小为2x2（即在2x2范围内取最大的特征）, 因此ksize=[1,2,2,1]
    # 步长为2, 因此strides=[1,2,2,1]
    # 一次两步则图像的长宽会减少一般, 相当于每两个像素点采样一次
    # padding采用的方式是SAME
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#############
## 构建模型 ##
############

# 模型结构:
# convolutional layer1 + max pooling;
# convolutional layer2 + max pooling;
# fully connected layer1 + dropout;
# fully connected layer2 to prediction.

xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1表示不确定, 由程序自动判断

# 第一层卷积层
# 卷积核大小为5×5, 即采样窗口大小为5x5
# 1是输入图片的chanel数（即深度）, 原始图片是灰度图, chanel只有1个
# 32是该层卷积核的个数, 也是输出的feature map层数（也就是下一卷积层所接收图片的chanel数）
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)  # 28×28×32
h_pool1 = max_pool_2x2(h_conv1)  # 14×14×32

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)  # 14×14×64
h_pool2 = max_pool_2x2(h_conv2)  # 7×7×64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 第一层全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# 优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#############
## 训练模型 ##
############

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 使用 Google 提供的MNIST数据集
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

for step in range(500):
    # 一次训练只取出部分数据, 即 MiniBatch
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={
        xs: batch_xs,
        ys: batch_ys,
        keep_prob: 0.5
    })
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

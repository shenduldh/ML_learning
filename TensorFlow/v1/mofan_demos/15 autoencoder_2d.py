from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)


# 任务: 将mnist数据中的图片进行压缩和解压缩,
#      然后将图片压缩后产生的二维特征显示在二维坐标轴上,
#      最后对比各个图片在二维坐标轴上的分类情况


LEARNING_RATE = 0.01
TRAINING_EPOCHS = 10
BATCH_SIZE = 256

INPUT_SIZE = 784
ONE_OUPUT_SIZE = 256
TWO_OUPUT_SIZE = 128
THREE_OUPUT_SIZE = 64
FOUR_OUPUT_SIZE = 2

weights = {
    'encoder_one': tf.Variable(tf.random_normal([INPUT_SIZE, ONE_OUPUT_SIZE])),
    'encoder_two': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE, TWO_OUPUT_SIZE])),
    'encoder_three': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE, THREE_OUPUT_SIZE])),
    'encoder_four': tf.Variable(tf.random_normal([THREE_OUPUT_SIZE, FOUR_OUPUT_SIZE])),

    'decoder_one': tf.Variable(tf.random_normal([FOUR_OUPUT_SIZE, THREE_OUPUT_SIZE])),
    'decoder_two': tf.Variable(tf.random_normal([THREE_OUPUT_SIZE, TWO_OUPUT_SIZE])),
    'decoder_three': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE, ONE_OUPUT_SIZE])),
    'decoder_four': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE, INPUT_SIZE])),
}

biases = {
    'encoder_one': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE])),
    'encoder_two': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE])),
    'encoder_three': tf.Variable(tf.random_normal([THREE_OUPUT_SIZE])),
    'encoder_four': tf.Variable(tf.random_normal([FOUR_OUPUT_SIZE])),

    'decoder_one': tf.Variable(tf.random_normal([THREE_OUPUT_SIZE])),
    'decoder_two': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE])),
    'decoder_three': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE])),
    'decoder_four': tf.Variable(tf.random_normal([INPUT_SIZE])),
}


# 一般 encoder 和 decoder 用的激活函数相同
def encoder(x):
    layer_one = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_one']), biases['encoder_one']))
    layer_two = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_one, weights['encoder_two']), biases['encoder_two']))
    layer_three = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_two, weights['encoder_three']), biases['encoder_three']))
    # 不进行sigmoid, 使得取值落在(-∞, +∞)
    layer_four = tf.add(
        tf.matmul(layer_three, weights['encoder_four']), biases['encoder_four'])
    return layer_four


def decoder(x):
    layer_one = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_one']), biases['decoder_one']))
    layer_two = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_one, weights['decoder_two']), biases['decoder_two']))
    layer_three = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_two, weights['decoder_three']), biases['decoder_three']))
    layer_four = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_three, weights['decoder_four']), biases['decoder_four']))
    return layer_four


x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
encoder = encoder(x)
prediction = decoder(encoder)

cost = tf.reduce_mean(tf.pow(prediction-x, 2))
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


print(mnist.test.images.shape)
print(mnist.test.labels.shape)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_count = int(mnist.train.num_examples/BATCH_SIZE)

    for step in range(TRAINING_EPOCHS):
        for i in range(batch_count):
            batch_x, _ = mnist.train.next_batch(BATCH_SIZE)
            _, cost_ = sess.run([train, cost], feed_dict={
                x: batch_x
            })

        print('cost:', cost_)

    labels = np.argmax(mnist.test.labels, axis=1)
    encoded_value = sess.run(
        encoder, feed_dict={x: mnist.test.images})
    plt.scatter(encoded_value[:, 0], encoded_value[:,
                1], c=labels)
    plt.show()

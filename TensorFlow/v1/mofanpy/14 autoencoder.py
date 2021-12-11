from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)


# 任务: 将mnist数据中的图片进行压缩和解压缩,
#      然后将解压缩后的图片与原始图片进行对比显示


LEARNING_RATE = 0.01
TRAINING_EPOCHS = 5
BATCH_SIZE = 256

INPUT_SIZE = 784
ONE_OUPUT_SIZE = 256
TWO_OUPUT_SIZE = 128


weights = {
    'encoder_one': tf.Variable(tf.random_normal([INPUT_SIZE, ONE_OUPUT_SIZE])),
    'encoder_two': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE, TWO_OUPUT_SIZE])),

    'decoder_one': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE, ONE_OUPUT_SIZE])),
    'decoder_two': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE, INPUT_SIZE])),
}

biases = {
    'encoder_one': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE])),
    'encoder_two': tf.Variable(tf.random_normal([TWO_OUPUT_SIZE])),

    'decoder_one': tf.Variable(tf.random_normal([ONE_OUPUT_SIZE])),
    'decoder_two': tf.Variable(tf.random_normal([INPUT_SIZE])),
}


# 一般 encoder 和 decoder 用的激活函数相同
def encoder(x):
    layer_one = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_one']), biases['encoder_one']))
    layer_two = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_one, weights['encoder_two']), biases['encoder_two']))
    return layer_two


def decoder(x):
    layer_one = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_one']), biases['decoder_one']))
    layer_two = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_one, weights['decoder_two']), biases['decoder_two']))
    return layer_two


x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
encoder_value = encoder(x)
prediction = decoder(encoder_value)

cost = tf.reduce_mean(tf.pow(prediction-x, 2))
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


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

    prediction_value = sess.run(
        prediction, feed_dict={x: mnist.test.images[:10]})
    # 创建一个具有2×10个网格的figure, 整个figure的尺寸为10×2
    _, axis = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axis[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        axis[1][i].imshow(np.reshape(prediction_value[i], (28, 28)))
    plt.show()

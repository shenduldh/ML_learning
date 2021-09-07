import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#########################
## 目标: 预测图片中的数字 ##
########################


def add_layer(inputs, in_size, out_size, activator=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    # 通过dropout将Wx_plus_b中的部分神经元的输出变成0
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activator is None:
        outputs = Wx_plus_b
    else:
        outputs = activator(Wx_plus_b)
    return outputs


#############
## 构建模型 ##
############
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8×8
ys = tf.placeholder(tf.float32, [None, 10])

layer_one = add_layer(xs, 64, 50, activator=tf.nn.tanh)

prediction = add_layer(layer_one, 50, 10, activator=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), axis=1))
tf.summary.scalar('loss', cross_entropy)

train = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)


#############
## 训练模型 ##
############

sess = tf.Session()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())


# 使用 sklearn 提供的数据集
# 数据集介绍:
#   输入: 8×8的手写数字图片
#   输出: 数字1-10的概率分布
#   标签: 数字1-10的 one_hot_vector
digits = load_digits()  # 加载数据集
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)  # 将label转换为one-hot vector的形式
# 分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

for step in range(500):
    sess.run(train, feed_dict={
        xs: X_train,
        ys: y_train,
        keep_prob: 1,
    })
    if step % 50 == 0:
        # merged也是此刻才计算出来的, 需要feed_dict
        train_loss = sess.run(merged, feed_dict={
            xs: X_train,
            ys: y_train,
            keep_prob: 1,
        })
        test_loss = sess.run(merged, feed_dict={
            xs: X_test,
            ys: y_test,
            keep_prob: 1,
        })
        train_writer.add_summary(train_loss, step)
        test_writer.add_summary(test_loss, step)
